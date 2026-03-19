import os
from contextlib import nullcontext
from copy import deepcopy
from logging import Logger
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torchinfo import summary

from data_processing.transforms import make_inverse_transforms, make_transforms
from data_processing.utils import init_data
from vae.models.predictor.joint_fm_latent_predictor import JointFMLatentPredictor
from vae.models.predictor.train import (
    get_rae_decoder,
    normalize_objective,
    save,
    update_ema,
    visualize_trajectory_rae,
)
from vae.models.predictor.utilities.ddp import cleanup, setup_ddp
from vae.models.predictor.utilities.log import create_logger
from vae.models.predictor.utilities.scheduler import LinearWarmupScheduler
from vae.models.predictor.utilities.seed import seed_everything


def _extract_batch(batch, device: torch.device, dtype=torch.float32):
    action_mask = None
    if isinstance(batch, (list, tuple)):
        first = batch[0]
        if isinstance(first, dict):
            x = first["visual"].to(device=device, dtype=dtype, non_blocking=True)
            proprio = first.get("proprio", None)
            if proprio is not None:
                proprio = proprio.to(device=device, dtype=dtype, non_blocking=True)
            if "action_mask" in first and torch.is_tensor(first["action_mask"]):
                action_mask = first["action_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
        else:
            x = first.to(device=device, dtype=dtype, non_blocking=True)
            proprio = None

        action = None
        if len(batch) > 1 and torch.is_tensor(batch[1]):
            action = batch[1].to(device=device, dtype=dtype, non_blocking=True)
    else:
        x = batch.to(device=device, dtype=dtype, non_blocking=True)
        action, proprio = None, None

    return x, action, proprio, action_mask


def _build_noisy_fm_inputs(raw_model, z, action, proprio, device, action_mask=None):
    n_ctx = raw_model.context_frames
    h_pred = raw_model.prediction_frames

    b, t, h, w, d = z.shape
    if t < (n_ctx + h_pred):
        raise ValueError(
            f"Sequence length {t} is too short for context={n_ctx} and pred={h_pred}."
        )

    z_ctx = z[:, :n_ctx, ...]
    z_tgt = z[:, n_ctx : n_ctx + h_pred, ...]

    if action is None:
        raise ValueError("Joint FM requires action tensor from dataset.")
    if action.size(1) < (n_ctx + h_pred):
        raise ValueError(
            f"Action sequence length {action.size(1)} is too short for context={n_ctx} and pred={h_pred}."
        )

    action_tgt = action[:, n_ctx : n_ctx + h_pred, :]

    tau_action = torch.rand((b,), device=device)
    tau_state = torch.rand((b,), device=device)
    if action_mask is not None:
        action_mask = action_mask.reshape(-1).to(device=device, dtype=torch.bool)
        if action_mask.shape[0] != b:
            raise ValueError(
                f"action_mask batch ({action_mask.shape[0]}) != data batch ({b})"
            )
        # UWM-style missing action handling: place missing actions at max noise timestep.
        tau_action = torch.where(action_mask, tau_action, torch.ones_like(tau_action))

    eps_action = torch.randn_like(action_tgt)
    eps_state = torch.randn_like(z_tgt)

    tau_action_view = tau_action.view(b, 1, 1)
    tau_state_view = tau_state.view(b, 1, 1, 1, 1)

    action_noisy = (1 - tau_action_view) * action_tgt + tau_action_view * eps_action
    z_noisy = (1 - tau_state_view) * z_tgt + tau_state_view * eps_state

    action_target_v = eps_action - action_tgt
    state_target_v = eps_state - z_tgt

    proprio_noisy = None
    proprio_ctx = None
    proprio_target_v = None
    if proprio is not None and getattr(raw_model.transformer, "has_proprio", False):
        if proprio.size(1) < (n_ctx + h_pred):
            raise ValueError(
                f"Proprio sequence length {proprio.size(1)} is too short for context={n_ctx} and pred={h_pred}."
            )
        proprio_ctx = proprio[:, :n_ctx, :]
        proprio_tgt = proprio[:, n_ctx : n_ctx + h_pred, :]
        eps_proprio = torch.randn_like(proprio_tgt)
        tau_prop_view = tau_state.view(b, 1, 1)
        proprio_noisy = (1 - tau_prop_view) * proprio_tgt + tau_prop_view * eps_proprio
        proprio_target_v = eps_proprio - proprio_tgt

    return {
        "z_ctx": z_ctx,
        "z_noisy": z_noisy,
        "action_noisy": action_noisy,
        "proprio_noisy": proprio_noisy,
        "proprio_ctx": proprio_ctx,
        "action_target_v": action_target_v,
        "state_target_v": state_target_v,
        "proprio_target_v": proprio_target_v,
        "tau_action": tau_action,
        "tau_state": tau_state,
    }


def training_step_joint(
    velocity_module: DDP,
    batch,
    mixed_precision: bool,
    device: torch.device,
    cfg: DictConfig,
):
    raw_model = velocity_module.module if isinstance(velocity_module, DDP) else velocity_module

    x, action, proprio, action_mask = _extract_batch(batch, device=device, dtype=torch.float32)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
        with torch.no_grad():
            z = raw_model.encode(x) if hasattr(raw_model, "encode") else raw_model.preprocess(x)

        fm_inputs = _build_noisy_fm_inputs(
            raw_model, z, action, proprio, device, action_mask=action_mask
        )

        action_time = fm_inputs["tau_action"] if raw_model.uses_timestep else None
        state_time = fm_inputs["tau_state"] if raw_model.uses_timestep else None

        action_pred, state_pred, proprio_pred = velocity_module(
            fm_inputs["z_ctx"],
            fm_inputs["z_noisy"],
            fm_inputs["action_noisy"],
            action_time=action_time,
            next_state_time=state_time,
            proprio_noisy=fm_inputs["proprio_noisy"],
            proprio_ctx=fm_inputs["proprio_ctx"],
            mask_action_from_visual=True,
        )

        loss_action = F.mse_loss(action_pred, fm_inputs["action_target_v"])
        loss_state = F.mse_loss(state_pred, fm_inputs["state_target_v"])

        loss_proprio = torch.tensor(0.0, device=device)
        if proprio_pred is not None and fm_inputs["proprio_target_v"] is not None:
            loss_proprio = F.mse_loss(proprio_pred, fm_inputs["proprio_target_v"])

        w_action = float(cfg.model.get("action_loss_weight", 1.0))
        w_state = float(cfg.model.get("state_loss_weight", 1.0))
        w_proprio = float(cfg.model.get("proprio_loss_weight", 1.0))

        total_loss = w_action * loss_action + w_state * loss_state + w_proprio * loss_proprio

    loss_info = {
        "loss": total_loss.detach(),
        "action_loss": loss_action.detach(),
        "state_loss": loss_state.detach(),
        "proprio_loss": loss_proprio.detach(),
    }
    return total_loss, loss_info


@torch.inference_mode()
def sample_joint_fm(
    velocity,
    z_ctx,
    proprio=None,
    steps=20,
    method="euler",
    ode_dtype=torch.float32,
):
    model_dtype = next(velocity.parameters()).dtype
    device_type = z_ctx.device.type

    def model_autocast_ctx():
        if device_type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
            return torch.amp.autocast(device_type="cuda", enabled=True, dtype=model_dtype)
        return nullcontext()

    amp_off = (
        torch.amp.autocast(device_type="cuda", enabled=False)
        if device_type == "cuda"
        else nullcontext()
    )

    with amp_off:
        z_ctx_ode = z_ctx.to(dtype=ode_dtype)
        z_ctx_model = z_ctx_ode.to(dtype=model_dtype)

        b, _, h, w, d = z_ctx_ode.shape
        h_pred = velocity.prediction_frames
        n_ctx = velocity.context_frames

        action_dim = int(getattr(velocity.transformer, "action_dim", 0))
        if action_dim <= 0:
            raise ValueError("Joint FM sampling requires action_dim > 0.")

        a_t = torch.randn((b, h_pred, action_dim), device=z_ctx_ode.device, dtype=ode_dtype)
        z_t = torch.randn((b, h_pred, h, w, d), device=z_ctx_ode.device, dtype=ode_dtype)

        use_proprio = bool(getattr(velocity.transformer, "has_proprio", False))
        if use_proprio:
            p_dim = int(getattr(velocity.transformer, "proprio_dim", 0))
            if torch.is_tensor(proprio):
                proprio_ode = proprio.to(device=z_ctx_ode.device, dtype=ode_dtype)
                if proprio_ode.ndim != 3 or proprio_ode.shape[-1] != p_dim:
                    raise ValueError(
                        f"Expected proprio shape [B, T, {p_dim}], got {tuple(proprio_ode.shape)}"
                    )
                if proprio_ode.shape[1] < n_ctx:
                    raise ValueError(
                        f"Proprio length {proprio_ode.shape[1]} is shorter than context_frames={n_ctx}."
                    )
                p_ctx = proprio_ode[:, :n_ctx, :]
            else:
                p_ctx = torch.zeros(
                    (b, n_ctx, p_dim),
                    device=z_ctx_ode.device,
                    dtype=ode_dtype,
                )
            p_t = torch.randn((b, h_pred, p_dim), device=z_ctx_ode.device, dtype=ode_dtype)

        tau_grid = torch.linspace(1.0, 0.0, steps, device=z_ctx_ode.device, dtype=ode_dtype)

        def v_joint(tau, state):
            if use_proprio:
                a_curr, z_curr, p_curr = state
            else:
                a_curr, z_curr = state
                p_curr = None

            if isinstance(tau, torch.Tensor):
                t_in = tau.expand(b).to(dtype=ode_dtype)
            else:
                t_in = torch.tensor([tau], device=z_ctx_ode.device, dtype=ode_dtype).expand(b)

            action_time = t_in.to(dtype=model_dtype) if velocity.uses_timestep else None
            state_time = t_in.to(dtype=model_dtype) if velocity.uses_timestep else None

            with model_autocast_ctx():
                proprio_input = None
                if p_curr is not None:
                    proprio_input = p_curr.to(dtype=model_dtype)
                out = velocity(
                    z_ctx_model,
                    z_curr.to(dtype=model_dtype),
                    a_curr.to(dtype=model_dtype),
                    action_time=action_time,
                    next_state_time=state_time,
                    proprio_noisy=proprio_input,
                    proprio_ctx=p_ctx.to(dtype=model_dtype) if p_curr is not None else None,
                    mask_action_from_visual=True,
                )

            a_v, z_v, p_v = out
            if use_proprio:
                if p_v is None:
                    p_v = torch.zeros_like(p_curr)
                else:
                    p_v = p_v.to(dtype=ode_dtype)
                return a_v.to(dtype=ode_dtype), z_v.to(dtype=ode_dtype), p_v

            return a_v.to(dtype=ode_dtype), z_v.to(dtype=ode_dtype)

        if use_proprio:
            a_traj, z_traj, _ = odeint(v_joint, (a_t, z_t, p_t), tau_grid, method=method)
        else:
            a_traj, z_traj = odeint(v_joint, (a_t, z_t), tau_grid, method=method)

        return z_traj[-1], a_traj[-1], z_traj


@torch.inference_mode()
def sample_policy_action_fm(
    velocity,
    z_ctx,
    proprio=None,
    steps=20,
    method="euler",
    ode_dtype=torch.float32,
    fixed_state_time=1.0,
):
    """
    UWM-like policy mode:
      - next-state branch time is fixed to high-noise end (continuous analog of t=T),
      - keep a fixed random noised next-state input,
      - denoise action branch only.
    """
    model_dtype = next(velocity.parameters()).dtype
    device_type = z_ctx.device.type

    def model_autocast_ctx():
        if device_type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
            return torch.amp.autocast(device_type="cuda", enabled=True, dtype=model_dtype)
        return nullcontext()

    amp_off = (
        torch.amp.autocast(device_type="cuda", enabled=False)
        if device_type == "cuda"
        else nullcontext()
    )

    with amp_off:
        z_ctx_ode = z_ctx.to(dtype=ode_dtype)
        z_ctx_model = z_ctx_ode.to(dtype=model_dtype)

        b, _, h, w, d = z_ctx_ode.shape
        h_pred = velocity.prediction_frames
        n_ctx = velocity.context_frames
        action_dim = int(getattr(velocity.transformer, "action_dim", 0))
        if action_dim <= 0:
            raise ValueError("Policy mode requires action_dim > 0.")

        # Action is denoised; next-state is fixed random noisy condition.
        a_t = torch.randn((b, h_pred, action_dim), device=z_ctx_ode.device, dtype=ode_dtype)
        z_fixed = torch.randn((b, h_pred, h, w, d), device=z_ctx_ode.device, dtype=ode_dtype)
        z_fixed_model = z_fixed.to(dtype=model_dtype)
        use_proprio = bool(getattr(velocity.transformer, "has_proprio", False))
        proprio_ctx_model = None
        proprio_fixed_model = None
        if use_proprio:
            p_dim = int(getattr(velocity.transformer, "proprio_dim", 0))
            if torch.is_tensor(proprio):
                proprio_ode = proprio.to(device=z_ctx_ode.device, dtype=ode_dtype)
                if proprio_ode.ndim != 3 or proprio_ode.shape[-1] != p_dim:
                    raise ValueError(
                        f"Expected proprio shape [B, T, {p_dim}], got {tuple(proprio_ode.shape)}"
                    )
                if proprio_ode.shape[1] < n_ctx:
                    raise ValueError(
                        f"Proprio length {proprio_ode.shape[1]} is shorter than context_frames={n_ctx}."
                    )
                p_ctx = proprio_ode[:, :n_ctx, :]
            else:
                p_ctx = torch.zeros(
                    (b, n_ctx, p_dim),
                    device=z_ctx_ode.device,
                    dtype=ode_dtype,
                )
            p_fixed = torch.randn((b, h_pred, p_dim), device=z_ctx_ode.device, dtype=ode_dtype)
            proprio_ctx_model = p_ctx.to(dtype=model_dtype)
            proprio_fixed_model = p_fixed.to(dtype=model_dtype)

        tau_grid = torch.linspace(1.0, 0.0, steps, device=z_ctx_ode.device, dtype=ode_dtype)
        state_time_fixed = torch.full(
            (b,),
            float(fixed_state_time),
            device=z_ctx_ode.device,
            dtype=ode_dtype,
        )

        def v_action(tau, a_curr):
            if isinstance(tau, torch.Tensor):
                t_in = tau.expand(b).to(dtype=ode_dtype)
            else:
                t_in = torch.tensor([tau], device=z_ctx_ode.device, dtype=ode_dtype).expand(b)

            action_time = t_in.to(dtype=model_dtype) if velocity.uses_timestep else None
            state_time = state_time_fixed.to(dtype=model_dtype) if velocity.uses_timestep else None

            with model_autocast_ctx():
                a_v, _, _ = velocity(
                    z_ctx_model,
                    z_fixed_model,
                    a_curr.to(dtype=model_dtype),
                    action_time=action_time,
                    next_state_time=state_time,
                    proprio_noisy=proprio_fixed_model,
                    proprio_ctx=proprio_ctx_model,
                    mask_action_from_visual=True,
                )
            return a_v.to(dtype=ode_dtype)

        a_traj = odeint(v_action, a_t, tau_grid, method=method)
        return a_traj[-1], a_traj


@torch.inference_mode()
def sample_forward_dynamics_fm(
    velocity,
    z_ctx,
    action,
    proprio=None,
    steps=20,
    method="euler",
    ode_dtype=torch.float32,
    fixed_action_time=0.0,
):
    """
    UWM-like forward dynamics mode:
      - action branch time is fixed at clean end (continuous analog of t=0),
      - action is treated as given condition (not denoised),
      - denoise next-state branch only.
    Args:
        action: [B, prediction_frames, action_dim], GT action for the prediction window.
    """
    model_dtype = next(velocity.parameters()).dtype
    device_type = z_ctx.device.type

    def model_autocast_ctx():
        if device_type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
            return torch.amp.autocast(device_type="cuda", enabled=True, dtype=model_dtype)
        return nullcontext()

    amp_off = (
        torch.amp.autocast(device_type="cuda", enabled=False)
        if device_type == "cuda"
        else nullcontext()
    )

    with amp_off:
        z_ctx_ode = z_ctx.to(dtype=ode_dtype)
        z_ctx_model = z_ctx_ode.to(dtype=model_dtype)

        b, _, h, w, d = z_ctx_ode.shape
        h_pred = velocity.prediction_frames
        n_ctx = velocity.context_frames
        action_dim = int(getattr(velocity.transformer, "action_dim", 0))
        if action_dim <= 0:
            raise ValueError("Forward dynamics mode requires action_dim > 0.")

        action_pred_in = action.to(dtype=ode_dtype)
        if action_pred_in.ndim != 3 or action_pred_in.shape[-1] != action_dim:
            raise ValueError(
                f"Expected action shape [B, {h_pred}, {action_dim}], got {tuple(action_pred_in.shape)}"
            )
        if action_pred_in.shape[1] != h_pred:
            raise ValueError(
                f"Expected action prediction window length {h_pred}, got {action_pred_in.shape[1]}."
            )
        action_pred_model = action_pred_in.to(dtype=model_dtype)
        use_proprio = bool(getattr(velocity.transformer, "has_proprio", False))

        z_t = torch.randn((b, h_pred, h, w, d), device=z_ctx_ode.device, dtype=ode_dtype)
        tau_grid = torch.linspace(1.0, 0.0, steps, device=z_ctx_ode.device, dtype=ode_dtype)
        action_time_fixed = torch.full(
            (b,),
            float(fixed_action_time),
            device=z_ctx_ode.device,
            dtype=ode_dtype,
        )

        if use_proprio:
            p_dim = int(getattr(velocity.transformer, "proprio_dim", 0))
            if torch.is_tensor(proprio):
                proprio_ode = proprio.to(device=z_ctx_ode.device, dtype=ode_dtype)
                if proprio_ode.ndim != 3 or proprio_ode.shape[-1] != p_dim:
                    raise ValueError(
                        f"Expected proprio shape [B, T, {p_dim}], got {tuple(proprio_ode.shape)}"
                    )
                if proprio_ode.shape[1] < n_ctx:
                    raise ValueError(
                        f"Proprio length {proprio_ode.shape[1]} is shorter than context_frames={n_ctx}."
                    )
                p_ctx = proprio_ode[:, :n_ctx, :]
            else:
                p_ctx = torch.zeros(
                    (b, n_ctx, p_dim),
                    device=z_ctx_ode.device,
                    dtype=ode_dtype,
                )
            p_t = torch.randn((b, h_pred, p_dim), device=z_ctx_ode.device, dtype=ode_dtype)

            def v_state(tau, state):
                z_curr, p_curr = state
                if isinstance(tau, torch.Tensor):
                    t_in = tau.expand(b).to(dtype=ode_dtype)
                else:
                    t_in = torch.tensor([tau], device=z_ctx_ode.device, dtype=ode_dtype).expand(b)

                action_time = (
                    action_time_fixed.to(dtype=model_dtype) if velocity.uses_timestep else None
                )
                state_time = t_in.to(dtype=model_dtype) if velocity.uses_timestep else None
                proprio_input = p_curr.to(dtype=model_dtype)

                with model_autocast_ctx():
                    _, z_v, p_v = velocity(
                        z_ctx_model,
                        z_curr.to(dtype=model_dtype),
                        action_pred_model,
                        action_time=action_time,
                        next_state_time=state_time,
                        proprio_noisy=proprio_input,
                        proprio_ctx=p_ctx.to(dtype=model_dtype),
                    )
                if p_v is None:
                    p_v = torch.zeros_like(p_curr)
                else:
                    p_v = p_v.to(dtype=ode_dtype)
                return z_v.to(dtype=ode_dtype), p_v

            z_traj, _ = odeint(v_state, (z_t, p_t), tau_grid, method=method)
            return z_traj[-1], z_traj

        def v_state(tau, z_curr):
            if isinstance(tau, torch.Tensor):
                t_in = tau.expand(b).to(dtype=ode_dtype)
            else:
                t_in = torch.tensor([tau], device=z_ctx_ode.device, dtype=ode_dtype).expand(b)

            action_time = action_time_fixed.to(dtype=model_dtype) if velocity.uses_timestep else None
            state_time = t_in.to(dtype=model_dtype) if velocity.uses_timestep else None

            with model_autocast_ctx():
                _, z_v, _ = velocity(
                    z_ctx_model,
                    z_curr.to(dtype=model_dtype),
                    action_pred_model,
                    action_time=action_time,
                    next_state_time=state_time,
                    proprio_noisy=None,
                    proprio_ctx=None,
                )
            return z_v.to(dtype=ode_dtype)

        z_traj = odeint(v_state, z_t, tau_grid, method=method)
        return z_traj[-1], z_traj


def evaluation_step_joint(velocity_module, batch, device, eval_steps=10):
    raw_model = velocity_module.module if isinstance(velocity_module, DDP) else velocity_module
    model_dtype = next(raw_model.parameters()).dtype

    x, action, proprio, _ = _extract_batch(batch, device=device, dtype=torch.float32)

    with torch.amp.autocast("cuda", enabled=True, dtype=model_dtype):
        with torch.inference_mode():
            z = raw_model.encode(x) if hasattr(raw_model, "encode") else raw_model.preprocess(x)
            z = z.float()

            n_ctx = raw_model.context_frames
            h_pred = raw_model.prediction_frames

            z_ctx = z[:, :n_ctx, ...]
            z_tgt = z[:, n_ctx : n_ctx + h_pred, ...]

            num_samples = 3
            z_preds = []
            for _ in range(num_samples):
                if action is not None:
                    if action.size(1) < (n_ctx + h_pred):
                        raise ValueError(
                            f"Action sequence length {action.size(1)} is too short for context={n_ctx} and pred={h_pred}."
                        )
                    action_pred = action[:, n_ctx : n_ctx + h_pred, :]
                    z_pred, _ = sample_forward_dynamics_fm(
                        raw_model,
                        z_ctx,
                        action_pred,
                        proprio=proprio,
                        steps=eval_steps,
                    )
                else:
                    z_pred, _, _ = sample_joint_fm(
                        raw_model,
                        z_ctx,
                        proprio=proprio,
                        steps=eval_steps,
                    )
                z_preds.append(z_pred.float())

            # [B, K, pred, h, w, d]
            z_preds = torch.stack(z_preds, dim=1)

            # Same metrics as original CFM branch.
            z_mean_pred = z_preds.mean(dim=1)
            mean_future_mse = F.mse_loss(z_mean_pred, z_tgt.float())

            z_tgt_rep = z_tgt.unsqueeze(1).expand(-1, num_samples, -1, -1, -1, -1)
            per_sample_mse = F.mse_loss(z_preds, z_tgt_rep, reduction="none")
            per_sample_mse = per_sample_mse.mean(dim=[2, 3, 4, 5])  # [B, K]
            min_of_k_mse = per_sample_mse.min(dim=1).values.mean(dim=0)
            mean_of_k_mse = per_sample_mse.mean(dim=1).mean(dim=0)

    return mean_future_mse, min_of_k_mse, mean_of_k_mse


def evaluate_joint(
    model,
    val_dataloader: DataLoader,
    device: torch.device,
    eval_steps: int = 10,
    max_batches: int | None = None,
):
    num_examples = torch.tensor(0.0, device=device)
    running_future_mse = torch.tensor(0.0, device=device)
    running_min_of_k_mse = torch.tensor(0.0, device=device)
    running_mean_of_k_mse = torch.tensor(0.0, device=device)
    processed_batches = 0

    for batch in val_dataloader:
        if max_batches is not None and processed_batches >= max_batches:
            break

        if isinstance(batch, (list, tuple)):
            if isinstance(batch[0], dict):
                bsz = batch[0]["visual"].shape[0]
            else:
                bsz = batch[0].shape[0]
        else:
            bsz = batch.shape[0]

        mean_future_mse, min_of_k_mse, mean_of_k_mse = evaluation_step_joint(
            model,
            batch,
            device,
            eval_steps=eval_steps,
        )

        running_future_mse += mean_future_mse * bsz
        running_min_of_k_mse += min_of_k_mse * bsz
        running_mean_of_k_mse += mean_of_k_mse * bsz
        num_examples += bsz
        processed_batches += 1

    packed = torch.stack(
        [
            running_future_mse,
            running_min_of_k_mse,
            running_mean_of_k_mse,
            num_examples,
        ]
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    final_future_mse = packed[0] / packed[3]
    final_min_of_k_mse = packed[1] / packed[3]
    final_mean_of_k_mse = packed[2] / packed[3]
    return final_future_mse, final_min_of_k_mse, final_mean_of_k_mse


@torch.inference_mode()
def plot_samples_joint(model, batch, device, cfg: DictConfig, sample_steps: int = 20):
    raw_model = model.module if hasattr(model, "module") else model
    model_dtype = next(raw_model.parameters()).dtype
    data_dtype = torch.float32

    n_ctx = raw_model.context_frames
    h_pred = raw_model.prediction_frames
    inv_transform = make_inverse_transforms(img_size=224)
    rae_model = get_rae_decoder(cfg, device)
    remove_dino_affine_for_rae = bool(cfg.vae.get("remove_dino_norm_affine_for_rae", True))

    x, action, proprio, _ = _extract_batch(batch, device=device, dtype=data_dtype)
    if action is None:
        raise ValueError("Visualization in forward-dynamics mode requires GT action.")

    with torch.amp.autocast("cuda", enabled=True, dtype=model_dtype):
        dino_features = raw_model.vae.preprocess(x)
        gt_dino_visual = dino_features[0, n_ctx : n_ctx + h_pred, ...]

        z_latent = raw_model.encode(x).to(dtype=data_dtype)
        z_ctx_0 = z_latent[0:1, :n_ctx, ...]
        if action.size(1) < (n_ctx + h_pred):
            raise ValueError(
                f"Action sequence length {action.size(1)} is too short for context={n_ctx} and pred={h_pred}."
            )
        action_0 = action[0:1, n_ctx : n_ctx + h_pred, :]
        proprio_0 = proprio[0:1] if proprio is not None else None

        batch_0_preds_latent = []
        for _ in range(3):
            # Visualization uses forward-dynamics mode only:
            # condition on GT action and denoise next-state.
            z_pred_last, _ = sample_forward_dynamics_fm(
                raw_model,
                z_ctx_0,
                action_0,
                proprio=proprio_0,
                steps=sample_steps,
            )
            batch_0_preds_latent.append(z_pred_last[0].to(dtype=data_dtype))

        preds_dino_visual = []
        for p in batch_0_preds_latent:
            decoded = raw_model.decode(p.unsqueeze(0).to(dtype=model_dtype))[0]
            preds_dino_visual.append(decoded)

        gt_raw_frames = x[0, n_ctx : n_ctx + h_pred, ...].float().cpu()
        gt_rgb_images = torch.clamp(inv_transform(gt_raw_frames), 0, 1)

    amp_off = (
        torch.amp.autocast(device_type="cuda", enabled=False)
        if device.type == "cuda"
        else nullcontext()
    )

    with amp_off:
        gt_rae_input = raw_model.vae.convert_to_rae_space(
            gt_dino_visual.float(),
            target_dim=768,
            remove_dino_norm_affine=remove_dino_affine_for_rae,
        )
        gt_rae_decoded = torch.clamp(
            rae_model.decode(gt_rae_input.permute(0, 3, 1, 2).float()),
            0,
            1,
        )

        pred_rae_decoded = []
        for pred in preds_dino_visual:
            pred_rae_input = raw_model.vae.convert_to_rae_space(
                pred.float(),
                target_dim=768,
                remove_dino_norm_affine=remove_dino_affine_for_rae,
            )
            pred_img = torch.clamp(
                rae_model.decode(pred_rae_input.permute(0, 3, 1, 2).float()),
                0,
                1,
            )
            pred_rae_decoded.append(pred_img)

    pil_image = visualize_trajectory_rae(gt_rgb_images, gt_rae_decoded, pred_rae_decoded)
    return pil_image


def train_joint(
    model: DDP,
    ema,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | LinearWarmupScheduler,
    cfg: DictConfig,
    device: torch.device,
    logger: Logger,
    rank: int,
    world_size: int,
    initial_epoch: int,
    initial_step: int,
    checkpoint_path: Path,
    samples_path: Path,
):
    model.train()
    ema.eval()

    optimizer.zero_grad()
    step = initial_step

    transform = make_transforms(img_size=224)

    context_len = cfg.model.context_frames
    pred_len = cfg.model.prediction_frames

    use_relative_action = bool(cfg.data.get("relative_action", True))

    (
        _,
        _,
        _,
        _,
        train_dataloader,
        val_dataloader,
        dist_sampler,
        viz_val_loader,
    ) = init_data(
        data_paths=cfg.data.paths,
        batch_size=cfg.training.batch_size,
        num_workers=8,
        dataset_type="custom",
        frameskip=cfg.data.frameskip,
        normalize_action=bool(cfg.data.get("normalize_action", True)),
        relative_action=use_relative_action,
        num_hist=context_len,
        num_pred=pred_len,
        transform=transform,
        pin_mem=True,
        persistent_workers=True,
        rank=rank,
        world_size=world_size,
        n_rollout=None,
    )

    train_sampler = dist_sampler
    eval_steps = int(cfg.training.get("eval_fm_steps", 10))
    sample_steps = int(cfg.training.get("sample_fm_steps", 20))
    smoke_test = bool(cfg.training.get("smoke_test", False))
    smoke_train_batches = int(cfg.training.get("smoke_train_batches", 1))
    smoke_eval_batches = int(cfg.training.get("smoke_eval_batches", 1))
    smoke_ckpt_name = str(cfg.training.get("smoke_ckpt_name", "smoke_one_batch.pt"))

    if smoke_test and rank == 0:
        logger.info(
            "Smoke test enabled: train_batches=%d eval_batches=%d sample_steps=%d",
            smoke_train_batches,
            smoke_eval_batches,
            sample_steps,
        )

    for epoch in range(initial_epoch, cfg.training.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_dataloader:
            loss, loss_info = training_step_joint(
                model,
                batch,
                cfg.training.mixed_precision,
                device,
                cfg,
            )

            loss_to_backward = loss / cfg.training.accumulation_steps
            loss_to_backward.backward()

            should_step = (
                True
                if cfg.training.accumulation_steps == 1
                else ((step + 1) % cfg.training.accumulation_steps == 0)
            )

            if should_step:
                if "clip_grad_norm" in cfg.training:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.training.clip_grad_norm,
                    )
                optimizer.step()
                update_ema(ema, model.module)
                scheduler.step()
                optimizer.zero_grad()

            if step % cfg.training.log_every_steps == 0 and rank == 0:
                logger.info(
                    "[epoch=%04d, step=%08d] loss=%.8f action=%.8f state=%.8f proprio=%.8f"
                    % (
                        epoch,
                        step,
                        float(loss_info["loss"].item()),
                        float(loss_info["action_loss"].item()),
                        float(loss_info["state_loss"].item()),
                        float(loss_info["proprio_loss"].item()),
                    )
                )
                if cfg.training.wandb and wandb.run is not None:
                    current_lr = optimizer.param_groups[0]["lr"]
                    wandb.log(
                        {
                            "train_loss": float(loss_info["loss"].item()),
                            "train_action_loss": float(loss_info["action_loss"].item()),
                            "train_state_loss": float(loss_info["state_loss"].item()),
                            "train_proprio_loss": float(loss_info["proprio_loss"].item()),
                            "lr": current_lr,
                        },
                        step=step,
                    )

            step += 1

            # One-batch preflight: run eval + visualization + checkpoint, then exit.
            if smoke_test and (step - initial_step) >= smoke_train_batches:
                model.eval()
                eval_loader = viz_val_loader if viz_val_loader is not None else val_dataloader
                mean_future_mse, min_of_k_mse, mean_of_k_mse = evaluate_joint(
                    model.module,
                    eval_loader,
                    device,
                    eval_steps=eval_steps,
                    max_batches=smoke_eval_batches,
                )
                model.train()

                for batch_viz in eval_loader:
                    break

                if rank == 0:
                    logger.info(
                        "[Smoke Test] step=%d val_future_mse=%.6f val_min_of_k_mse=%.6f val_mean_of_k_mse=%.6f",
                        step,
                        mean_future_mse.item(),
                        min_of_k_mse.item(),
                        mean_of_k_mse.item(),
                    )
                    samples = plot_samples_joint(
                        model,
                        batch_viz,
                        device,
                        cfg,
                        sample_steps=sample_steps,
                    )
                    smoke_viz_dir = samples_path / "smoke_visuals"
                    smoke_viz_dir.mkdir(exist_ok=True, parents=True)
                    samples.save(smoke_viz_dir / f"step_{step}.jpg")

                    save(
                        checkpoint_path / smoke_ckpt_name,
                        model.module,
                        ema,
                        optimizer,
                        scheduler,
                        epoch,
                        step,
                        cfg,
                    )
                    logger.info(
                        "[Smoke Test] saved checkpoint=%s, visualization_dir=%s",
                        checkpoint_path / smoke_ckpt_name,
                        smoke_viz_dir,
                    )

                dist.barrier()
                return

            if step % 5000 == 0:
                if rank == 0:
                    save(
                        checkpoint_path / f"step_{step:08d}.pt",
                        model.module,
                        ema,
                        optimizer,
                        scheduler,
                        epoch,
                        step,
                        cfg,
                    )
                    logger.info(f"Checkpoint saved at step {step}")

                model.eval()
                eval_loader = viz_val_loader if viz_val_loader is not None else val_dataloader

                mean_future_mse, min_of_k_mse, mean_of_k_mse = evaluate_joint(
                    model.module,
                    eval_loader,
                    device,
                    eval_steps=eval_steps,
                )
                model.train()

                if rank == 0:
                    logger.info(
                        "[Step Evaluation] step=%d val_future_mse=%.6f val_min_of_k_mse=%.6f val_mean_of_k_mse=%.6f",
                        step,
                        mean_future_mse.item(),
                        min_of_k_mse.item(),
                        mean_of_k_mse.item(),
                    )
                    if cfg.training.wandb and wandb.run is not None:
                        wandb.log(
                            {
                                "val_mean_future_mse": mean_future_mse.item(),
                                "val_min_of_k_mse": min_of_k_mse.item(),
                                "val_mean_of_k_mse": mean_of_k_mse.item(),
                            },
                            step=step,
                        )

                for batch_viz in eval_loader:
                    break

                if rank == 0:
                    samples = plot_samples_joint(
                        model,
                        batch_viz,
                        device,
                        cfg,
                        sample_steps=sample_steps,
                    )
                    save_dir = samples_path / "step_visuals"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    samples.save(save_dir / f"step_{step}.jpg")

                    if cfg.training.wandb and wandb.run is not None:
                        wandb.log(
                            {
                                "val_samples": wandb.Image(samples),
                            },
                            step=step,
                        )

                dist.barrier()

        if rank == 0:
            save(
                checkpoint_path / "current.pt",
                model.module,
                ema,
                optimizer,
                scheduler,
                epoch,
                step,
                cfg,
            )

        dist.barrier()

        if epoch % cfg.training.ckpt_every_epochs == 0 and rank == 0:
            save(
                checkpoint_path / f"epoch={epoch:04d}.pt",
                model.module,
                ema,
                optimizer,
                scheduler,
                epoch,
                step,
                cfg,
            )

        dist.barrier()

        if epoch % cfg.training.evaluate_every_epochs == 0:
            model.eval()
            eval_loader = viz_val_loader if viz_val_loader is not None else val_dataloader
            mean_future_mse, min_of_k_mse, mean_of_k_mse = evaluate_joint(
                model.module,
                eval_loader,
                device,
                eval_steps=eval_steps,
            )
            model.train()

            if rank == 0:
                logger.info(
                    "[epoch=%d] val_future_mse=%.6f val_min_of_k_mse=%.6f val_mean_of_k_mse=%.6f",
                    epoch,
                    mean_future_mse.item(),
                    min_of_k_mse.item(),
                    mean_of_k_mse.item(),
                )
                if cfg.training.wandb and wandb.run is not None:
                    wandb.log(
                        {
                            "val_mean_future_mse": mean_future_mse.item(),
                            "val_min_of_k_mse": min_of_k_mse.item(),
                            "val_mean_of_k_mse": mean_of_k_mse.item(),
                        },
                        step=step,
                    )

        dist.barrier()

        if epoch % cfg.training.sample_every_epochs == 0:
            eval_loader = viz_val_loader if viz_val_loader is not None else val_dataloader
            for batch_viz in eval_loader:
                break

            if rank == 0:
                samples = plot_samples_joint(
                    model,
                    batch_viz,
                    device,
                    cfg,
                    sample_steps=sample_steps,
                )
                save_dir = samples_path / f"epoch_{epoch}"
                save_dir.mkdir(exist_ok=True, parents=True)
                samples.save(save_dir / f"step_{step}.jpg")

                if cfg.training.wandb and wandb.run is not None:
                    wandb.log(
                        {
                            "val_samples": wandb.Image(samples),
                        },
                        step=step,
                    )

        dist.barrier()


config_path = os.environ.get(
    "CONFIG_PATH",
    "/home/lizijia/Gen-JEPA/vae/models/configs/kubric",
)
config_name = os.environ.get(
    "CONFIG_NAME",
    "joint_fm.yaml",
)


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    rank, world_size, local_rank, device = setup_ddp()
    seed_everything(cfg.training.seed + rank)

    if rank == 0:
        print("--- Configuration ---")
        print(OmegaConf.to_yaml(cfg))
        print("---------------------")

    experiment_dir = Path(cfg.experiment_path) / cfg.name
    checkpoint_dir = experiment_dir / "checkpoints"
    samples_dir = experiment_dir / "samples"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)

    logger = create_logger(experiment_dir, rank)
    if rank == 0:
        logger.info(f"experiment directory created at {experiment_dir}")

    initial_epoch = 0
    initial_step = 0

    objective = normalize_objective(cfg.objective)
    if objective != "CFM":
        raise NotImplementedError(
            "train_joint_fm.py only supports generative FM objective (cfg.objective=CFM)."
        )
    if "vae" not in cfg:
        raise NotImplementedError("Only VAE-based predictor is supported.")
    if "fm_variant" not in cfg.model:
        raise ValueError(
            "Config mismatch: expected joint FM model config (missing cfg.model.fm_variant). "
            "Use CONFIG_NAME=joint_fm.yaml instead of dino_foresight_cfm."
        )

    torch.cuda.empty_cache()
    model = JointFMLatentPredictor(**cfg.model, vae=cfg.vae)

    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad = False

    model = model.to(device).bfloat16()
    ema = ema.to(device).bfloat16()
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    if rank == 0:
        logger.info(
            f"Joint FM model initialized. fm_variant={cfg.model.get('fm_variant', 'fm')}, "
            f"use_timestep={model.module.uses_timestep}"
        )
        try:
            summary(model.module.transformer, depth=3)
        except Exception as e:
            logger.warning(f"Could not print summary: {e}")

    if "resume" in cfg:
        ckpt = torch.load(cfg.resume.ckpt, map_location="cpu", weights_only=True)
        model.module.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        if rank == 0:
            logger.info(f"restored ckpt: {cfg.resume.ckpt}")
        initial_epoch = ckpt["epoch"] + 1
        initial_step = ckpt["step"] + 1
        del ckpt
        torch.cuda.empty_cache()

    if cfg.training.wandb and rank == 0:
        run_id = None
        if "resume" in cfg and cfg.resume.wandb:
            ckpt_meta = torch.load(cfg.resume.ckpt, map_location="cpu", weights_only=True)
            run_id = ckpt_meta.get("wandb", {}).get("run", {}).get("id", None)
            del ckpt_meta

        wandb.init(
            project="gen-jepa_vae",
            name=str(cfg.name),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow" if run_id else None,
            id=run_id,
        )

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)

    scheduler_name = str(cfg.training.lr_scheduler.name)
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=cfg.training.steps)
    elif scheduler_name == "linear_warmup":
        scheduler = LinearWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=cfg.training.lr_scheduler.warmup_steps,
            lr_max=cfg.training.lr,
        )
    else:
        raise NotImplementedError(f"Unsupported scheduler: {scheduler_name}")

    if "resume" in cfg:
        if rank == 0:
            logger.info(
                f"Reloading checkpoint for optimizer/scheduler state: {cfg.resume.ckpt}"
            )
        ckpt = torch.load(cfg.resume.ckpt, map_location="cpu", weights_only=True)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Failed to load optimizer state: {e}")
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        del ckpt
        torch.cuda.empty_cache()

    train_joint(
        model,
        ema,
        optimizer,
        scheduler,
        cfg,
        device,
        logger,
        rank,
        world_size,
        initial_epoch,
        initial_step,
        checkpoint_dir,
        samples_dir,
    )
    cleanup()


if __name__ == "__main__":
    main()
