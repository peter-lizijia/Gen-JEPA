import os
import ssl

try:
  _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
  pass
else:
  ssl._create_default_https_context = _create_unverified_https_context

for env_key in ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]:
  if env_key in os.environ:
    del os.environ[env_key]

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "True"

import io
import sys
from copy import deepcopy
from contextlib import nullcontext
from functools import partial
from logging import Logger
from pathlib import Path
from typing import OrderedDict, Optional, List

import einops
import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdiffeq import odeint
from torchinfo import summary

# Utilities
from vae.models.predictor.utilities.ddp import cleanup, setup_ddp
from vae.models.predictor.utilities.log import create_logger
from vae.models.predictor.utilities.scheduler import LinearWarmupScheduler
from vae.models.predictor.utilities.seed import seed_everything

# Models
from vae.models.predictor.autoregressive_dino_foresight_latent import (
    AutoregressiveDINOForesightLatent,
    AutoregressiveDINOForesightLatentRegressor,
)

# Data Processing
from data_processing.utils import init_data
from data_processing.transforms import make_transforms, make_inverse_transforms

# Visualization tools
import matplotlib.pyplot as plt
from PIL import Image

# [Optimization] reduce fragmentation in PyTorch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# [Optimization] use TF32 to accerlerate Ampere+ GPU
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PredictorModel = (
    AutoregressiveDINOForesightLatent | AutoregressiveDINOForesightLatentRegressor
)
REGRESSION_OBJECTIVES = {"REG", "L2"}
_RAE_DECODER_CACHE: dict[str, torch.nn.Module] = {}


def normalize_objective(objective: str) -> str:
  return str(objective).upper()


def is_regression_objective(objective: str) -> bool:
  return normalize_objective(objective) in REGRESSION_OBJECTIVES


def _resolve_hf_model_snapshot(repo_id: str) -> Optional[str]:
  # Convert "org/model" -> "~/.cache/huggingface/hub/models--org--model/snapshots/<hash>"
  repo_key = repo_id.replace("/", "--")
  snapshots_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_key}" / "snapshots"
  if not snapshots_root.is_dir():
    return None
  candidates = [p for p in snapshots_root.iterdir() if p.is_dir()]
  if not candidates:
    return None
  latest = max(candidates, key=lambda p: p.stat().st_mtime)
  return str(latest)


def get_rae_decoder(cfg: DictConfig, device: torch.device):
  if "vae" not in cfg:
    raise ValueError("RAE visualization requires cfg.vae.")

  vae_cfg = cfg.vae
  rae_config_path = "/home/lizijia/Gen-JEPA/vae/models/rae/configs/stage1/pretrained/DINOv2-B.yaml"
  rae_decoder_config_path = "/home/lizijia/Gen-JEPA/vae/models/rae/configs/decoder/ViTXL/config.json"
  rae_ckpt_path = "/home/lizijia/Gen-JEPA/vae/models/rae/models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    
  if rae_config_path is None or rae_decoder_config_path is None or rae_ckpt_path is None:
    raise ValueError(
        "Missing RAE config in cfg.vae. Expected keys: "
        "rae_config_path, rae_decoder_config_path, rae_ckpt_path"
    )

  cache_key = f"{device.type}:{device.index}:{rae_ckpt_path}"
  if cache_key in _RAE_DECODER_CACHE:
    return _RAE_DECODER_CACHE[cache_key]

  rae_src = Path(__file__).resolve().parents[1] / "rae" / "src"
  rae_src_str = str(rae_src)
  added_path = False
  if rae_src_str not in sys.path:
    sys.path.insert(0, rae_src_str)
    added_path = True

  try:
    from stage1 import RAE
  finally:
    if added_path and rae_src_str in sys.path:
      sys.path.remove(rae_src_str)

  rae_yaml_cfg = OmegaConf.load(rae_config_path)
  rae_params = OmegaConf.to_container(rae_yaml_cfg.stage_1.params, resolve=True)
  rae_params["decoder_config_path"] = rae_decoder_config_path

  encoder_cfg = rae_params.get("encoder_config_path", None)
  if isinstance(encoder_cfg, str) and not Path(encoder_cfg).exists():
    local_snapshot = _resolve_hf_model_snapshot(encoder_cfg)
    if local_snapshot is None:
      raise FileNotFoundError(
          f"Cannot resolve local HuggingFace snapshot for encoder '{encoder_cfg}'. "
          "Please ensure it is cached under ~/.cache/huggingface/hub first."
      )
    rae_params["encoder_config_path"] = local_snapshot

  model_cfg = vae_cfg.get("model", None)
  if model_cfg is not None:
    rae_stats_path = model_cfg.get("rae_stats_path", None)
    if rae_stats_path:
      rae_params["normalization_stat_path"] = rae_stats_path

  rae_model = RAE(**rae_params).to(device).eval()

  rae_st = torch.load(rae_ckpt_path, map_location=device)
  if isinstance(rae_st, dict) and "state_dict" in rae_st:
    rae_st = rae_st["state_dict"]

  decoder_st = {
      k.replace("decoder.", ""): v
      for k, v in rae_st.items()
      if isinstance(k, str) and k.startswith("decoder.")
  }
  rae_model.decoder.load_state_dict(decoder_st, strict=False)

  _RAE_DECODER_CACHE[cache_key] = rae_model
  return rae_model


def save(
    path: Path,
    model: PredictorModel,
    ema: PredictorModel,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | LinearWarmupScheduler,
    epoch: int,
    step: int,
    cfg: DictConfig
):
  wandb_info = {}
  if cfg.training.wandb and wandb.run is not None:
    wandb_info = {
        "wandb": {
            "run": {
                "id": wandb.run.id,
                "name": wandb.run.name
            }
        }
    }

  # unwrapped model (model.module)
  torch.save({
      "model_state_dict": model.state_dict(),
      "ema_state_dict": ema.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "scheduler_state_dict": scheduler.state_dict(),
      "epoch": epoch,
      "step": step,
      **wandb_info
  }, path)


def training_step(
        velocity_module: DDP,
        batch,
        mixed_precision,
        device):
  
  # get model parameters dynamically from the DDP-wrapped module
  raw_model = velocity_module.module if isinstance(velocity_module, DDP) else velocity_module
  n_ctx = raw_model.context_frames
  H_pred = raw_model.prediction_frames

  # get data
  if isinstance(batch, (list, tuple)):
    # extratct visual and proprio
    data_dict = batch[0]
    # non_blocking=True accelerates data transfer
    x = data_dict['visual'].to(device, non_blocking=True)  
    proprio = data_dict['proprio'].to(device, non_blocking=True)

    # extract other data
    action = batch[1].to(device, non_blocking=True) if len(batch) > 1 else None
  else:
    # for robustness
    x = batch.to(device, non_blocking=True)
    action, proprio = None, None
  
  # use bfloat16 to train with mixed precision
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
    # 2. Encode
    with torch.no_grad():
      if hasattr(raw_model, "encode"):
        # x: [B, T + H, c, h, w] -> z: [B, T + H, h', w', d]
        z = raw_model.encode(x)
      else:
        z = raw_model.preprocess(x)
    
    b, t, h, w, d = z.shape
    
    # make sure we have enough frames for context + prediction
    assert t >= (n_ctx + H_pred), f"Sequence length {t} is too short for {n_ctx} context + {H_pred} future"

    # construct Context and Target
    # Context: first n_ctx frames
    # Target:  last H_pred frames (training target)
    z_ctx = z[:, :n_ctx, ...] 
    z_tgt = z[:, n_ctx:n_ctx + H_pred, ...]
    
    # Flow Matching Setup (Targeting Future)
    # Sample time tau [0, 1]
    tau = torch.rand(size=[b, ], device=device)
    
    # Generate Noise for Future Visual Latents
    epsilon = torch.randn(b, H_pred, h, w, d, device=device)
    
    # Interpolation: z_tau = (1-t)*Data + t*Noise
    # Target vector field v = dx/dt = x_1 - x_0 = epsilon - z_tgt
    tau_view = tau.view(b, 1, 1, 1, 1)
    z_tau = (1 - tau_view) * z_tgt + tau_view * epsilon
    
    # Build FM input for proprio as well (generation path).
    proprio_input = proprio
    epsilon_prop = None
    p_tgt = None
    if proprio is not None:
        p_ctx = proprio[:, :n_ctx, :]
        p_tgt = proprio[:, n_ctx:n_ctx + H_pred, :]
        epsilon_prop = torch.randn_like(p_tgt)
        tau_prop = tau.view(b, 1, 1)
        p_tau = (1 - tau_prop) * p_tgt + tau_prop * epsilon_prop
        proprio_input = torch.cat([p_ctx, p_tau], dim=1)

    # forward
    # input: Context + Noisy Future (visual/proprio) + Condition(Action)
    model_output = velocity_module(
        z_ctx, 
        z_tau, 
        tau, 
        action=action, 
        proprio=proprio_input
    )
    
    if isinstance(model_output, tuple):
        v_vis, v_prop_pred = model_output
    else:
        v_vis, v_prop_pred = model_output, None
    
    # calculate Loss
    # A. Visual Loss (Flow Matching)
    # Target: epsilon - z_tgt
    target_v = epsilon - z_tgt
    loss_vis = F.mse_loss(v_vis, target_v)
    
    # B. Proprio Loss (Flow Matching)
    loss_prop = torch.tensor(0.0, device=device)
    if v_prop_pred is not None and epsilon_prop is not None and p_tgt is not None:
        target_v_prop = epsilon_prop - p_tgt
        loss_prop = F.mse_loss(v_prop_pred, target_v_prop)
        
    # total Loss
    total_loss = loss_vis + 0.1 * loss_prop
    
    return total_loss


def training_step_regression(
        velocity_module: DDP,
        batch,
        mixed_precision,
        device):

  raw_model = velocity_module.module if isinstance(velocity_module, DDP) else velocity_module
  n_ctx = raw_model.context_frames
  H_pred = raw_model.prediction_frames

  if isinstance(batch, (list, tuple)):
    data_dict = batch[0]
    x = data_dict['visual'].to(device, non_blocking=True)
    proprio = data_dict['proprio'].to(device, non_blocking=True)
    action = batch[1].to(device, non_blocking=True) if len(batch) > 1 else None
  else:
    x = batch.to(device, non_blocking=True)
    action, proprio = None, None

  with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
    with torch.no_grad():
      if hasattr(raw_model, "encode"):
        z = raw_model.encode(x)
      else:
        z = raw_model.preprocess(x)

    t = z.shape[1]
    assert t >= (n_ctx + H_pred), f"Sequence length {t} is too short for {n_ctx} context + {H_pred} future"

    z_ctx = z[:, :n_ctx, ...]
    z_tgt = z[:, n_ctx:n_ctx + H_pred, ...]

    model_output = velocity_module(
        z_ctx,
        action=action,
        proprio=proprio
    )

    if isinstance(model_output, tuple):
      pred_vis, pred_prop = model_output
    else:
      pred_vis, pred_prop = model_output, None

    loss_vis = F.mse_loss(pred_vis, z_tgt)

    loss_prop = torch.tensor(0.0, device=device)
    if pred_prop is not None and proprio is not None:
      p_tgt = proprio[:, n_ctx:n_ctx + H_pred, :]
      loss_prop = F.mse_loss(pred_prop, p_tgt)

    total_loss = loss_vis + 0.1 * loss_prop
    return total_loss


def evaluation_step(velocity_module, batch, device):
    # 1. model precision
    raw_model = velocity_module.module if isinstance(velocity_module, DDP) else velocity_module
    model_dtype = next(raw_model.parameters()).dtype
    
    n_ctx = raw_model.context_frames
    H_pred = raw_model.prediction_frames

    # get data
    if isinstance(batch, (list, tuple)):
        x = batch[0]
        if isinstance(x, dict):
            proprio = x['proprio'].to(device=device, dtype=model_dtype, non_blocking=True)
            x = x['visual'].to(device=device, dtype=model_dtype, non_blocking=True)
        else:
            x = x.to(device=device, dtype=model_dtype, non_blocking=True)
            proprio = None

        action = batch[1].to(device=device, dtype=model_dtype, non_blocking=True) if len(batch) > 1 else None
        if len(batch) > 2 and proprio is None:
            proprio = batch[2].to(device=device, dtype=model_dtype, non_blocking=True)
    else:
        x = batch.to(device=device, dtype=model_dtype, non_blocking=True)
        action, proprio = None, None

    # inference
    # use autocast explicitly for type match
    with torch.amp.autocast('cuda', enabled=True, dtype=model_dtype):
        with torch.inference_mode():
            # 2. encode
            if hasattr(raw_model, "encode"):
                z = raw_model.encode(x)
            else:
                z = raw_model.preprocess(x)
            
            z = z.to(model_dtype)

            # z shape: [B, T, H, W, D]
            z_ctx = z[:, :n_ctx, ...]
            z_tgt = z[:, n_ctx:n_ctx + H_pred, ...] # GT Future

            num_samples = 3
            num_steps = 10  
            
            # 3. Sample K times
            z_preds = []
            for k in range(num_samples):
                z_pred, _ = sample(
                    raw_model, 
                    z_ctx,
                    action,
                    proprio,
                    steps=num_steps
                )
                z_preds.append(z_pred.to(model_dtype))
            
            # Stack: [B, K, T, H, W, D]
            z_preds = torch.stack(z_preds, dim=1)
            
            # 4. metric (Mean Future MSE)
            z_mean_pred = torch.mean(z_preds, dim=1) # [B, T, H, W, D]
            mean_future_mse = F.mse_loss(z_mean_pred, z_tgt)
            
            # Best of K & Mean of K
            z_tgt_rep = einops.repeat(z_tgt, "b t h w d -> b k t h w d", k=num_samples)

            # MSE per sample: [B, K, T, H, W, D]
            per_sample_mse = F.mse_loss(z_preds, z_tgt_rep, reduction="none")
            # taking mean in spatial and temporal dimension -> [B, K]
            per_sample_mse = per_sample_mse.mean(dim=[2, 3, 4, 5]) 
            
            min_of_k_mse = per_sample_mse.min(dim=1).values.mean(dim=0)
            mean_of_k_mse = per_sample_mse.mean(dim=1).mean(dim=0)
            
    return (mean_future_mse, min_of_k_mse, mean_of_k_mse)


def evaluation_step_regression(velocity_module, batch, device):
    raw_model = velocity_module.module if isinstance(velocity_module, DDP) else velocity_module
    model_dtype = next(raw_model.parameters()).dtype

    n_ctx = raw_model.context_frames
    H_pred = raw_model.prediction_frames

    if isinstance(batch, (list, tuple)):
        x = batch[0]
        if isinstance(x, dict):
            proprio = x['proprio'].to(device=device, dtype=model_dtype, non_blocking=True)
            x = x['visual'].to(device=device, dtype=model_dtype, non_blocking=True)
        else:
            x = x.to(device=device, dtype=model_dtype, non_blocking=True)
            proprio = None

        action = batch[1].to(device=device, dtype=model_dtype, non_blocking=True) if len(batch) > 1 else None
        if len(batch) > 2 and proprio is None:
            proprio = batch[2].to(device=device, dtype=model_dtype, non_blocking=True)
    else:
        x = batch.to(device=device, dtype=model_dtype, non_blocking=True)
        action, proprio = None, None

    with torch.amp.autocast('cuda', enabled=True, dtype=model_dtype):
        with torch.inference_mode():
            if hasattr(raw_model, "encode"):
                z = raw_model.encode(x)
            else:
                z = raw_model.preprocess(x)
            z = z.to(model_dtype)

            z_ctx = z[:, :n_ctx, ...]
            z_tgt = z[:, n_ctx:n_ctx + H_pred, ...]

            model_output = raw_model(
                z_ctx,
                action=action,
                proprio=proprio
            )
            if isinstance(model_output, tuple):
                pred_vis, _ = model_output
            else:
                pred_vis = model_output

            mean_future_mse = F.mse_loss(pred_vis, z_tgt)

    # Keep output signature identical to CFM evaluation for logging compatibility.
    return (mean_future_mse, mean_future_mse, mean_future_mse)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
  ema_params = OrderedDict(ema_model.named_parameters())
  model_params = OrderedDict(model.named_parameters())
  for name, param in model_params.items():
    ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def evaluate(
    model: PredictorModel,
    val_dataloader: DataLoader,
    device: torch.device,
    objective: str = "CFM"
):
  num_examples = torch.tensor(0.0, device=device)
  running_future_mse = torch.tensor(0.0, device=device)
  running_min_of_k_mse = torch.tensor(0.0, device=device)
  running_mean_of_k_mse = torch.tensor(0.0, device=device)

  for batch in val_dataloader:
    # get batch size
    if isinstance(batch, (list, tuple)):
        if isinstance(batch[0], dict):
            b = batch[0]['visual'].shape[0]
        else:
            b = batch[0].shape[0]
    else:
        b = batch.shape[0]

    if is_regression_objective(objective):
      (mean_future_mse, min_of_k_mse, mean_of_k_mse) = evaluation_step_regression(
          model,
          batch,
          device
      )
    else:
      (mean_future_mse, min_of_k_mse, mean_of_k_mse) = evaluation_step(
          model,
          batch,
          device
      )
    
    # accumulate "Sum of MSEs" (Mean * Count)
    running_future_mse += mean_future_mse * b
    running_min_of_k_mse += min_of_k_mse * b
    running_mean_of_k_mse += mean_of_k_mse * b
    num_examples += b

  # DDP Sync
  packed_metrics = torch.stack([
      running_future_mse, 
      running_min_of_k_mse, 
      running_mean_of_k_mse, 
      num_examples
  ])
  
  # sum of all gpus
  dist.all_reduce(packed_metrics, op=dist.ReduceOp.SUM)
  
  # global mean
  global_count = packed_metrics[3]
  final_future_mse = packed_metrics[0] / global_count
  final_min_of_k_mse = packed_metrics[1] / global_count
  final_mean_of_k_mse = packed_metrics[2] / global_count

  return final_future_mse, final_min_of_k_mse, final_mean_of_k_mse


@torch.inference_mode()
def sample(
    velocity: PredictorModel,
    z_ctx,
    action=None,
    proprio=None,
    steps=50,
    method="euler"
):
  """
  sample: z_ctx (Context) + Noise -> ODE Integration -> z_pred (Future)
  Flow matching: from Noise to Data, so t=1 is noise, t=0 is data
  """
  if getattr(velocity, "is_regression_model", False):
    out = velocity.forward(
        z_ctx,
        action=action,
        proprio=proprio
    )
    if isinstance(out, tuple):
      pred_vis, _ = out
    else:
      pred_vis = out
    return pred_vis, pred_vis.unsqueeze(0)

  H_pred = velocity.prediction_frames
  n_ctx = velocity.context_frames
  
  b, _, h, w, d = z_ctx.shape
  
  # t=1: Pure Noise for visual state
  z_t = torch.randn((b, H_pred, h, w, d), device=z_ctx.device, dtype=z_ctx.dtype)

  # If proprio exists, integrate it jointly with visual during FM sampling.
  use_proprio_fm = proprio is not None
  if use_proprio_fm:
    p_ctx = proprio[:, :n_ctx, :]
    p_dim = p_ctx.shape[-1]
    p_t = torch.randn((b, H_pred, p_dim), device=z_ctx.device, dtype=z_ctx.dtype)
  
  # Time grid: 1.0 -> 0.0
  tau_grid = torch.linspace(1., 0., steps, device=z_ctx.device, dtype=z_ctx.dtype)

  def v(tau, z_curr):
    """
    ODE: dy/dt = v(y, t)
    """
    # broadcast tau to [B, ] for batch processing
    if isinstance(tau, torch.Tensor):
        t_in = tau.expand(b)
    else:
        t_in = torch.tensor([tau], device=z_ctx.device, dtype=z_ctx.dtype).expand(b)
        
    # use model
    out = velocity.forward(
        z_ctx,
        z_curr, # current visual state
        t_in,
        action=action,
        proprio=proprio
    )
    
    if isinstance(out, tuple):
        v_vis, _ = out
        return v_vis
    return out

  if not use_proprio_fm:
    # ODE integration (visual only)
    z_traj = odeint(v, z_t, tau_grid, method=method)
    return z_traj[-1], z_traj

  def v_joint(tau, state):
    z_curr, p_curr = state
    if isinstance(tau, torch.Tensor):
      t_in = tau.expand(b)
    else:
      t_in = torch.tensor([tau], device=z_ctx.device, dtype=z_ctx.dtype).expand(b)

    proprio_input = torch.cat([p_ctx, p_curr], dim=1)
    out = velocity.forward(
        z_ctx,
        z_curr,
        t_in,
        action=action,
        proprio=proprio_input
    )

    if isinstance(out, tuple):
      v_vis, v_prop = out
    else:
      v_vis, v_prop = out, None

    if v_prop is None:
      v_prop = torch.zeros_like(p_curr)

    return v_vis, v_prop

  # ODE integration (visual + proprio). Return visual trajectory for existing callers.
  z_traj, _ = odeint(v_joint, (z_t, p_t), tau_grid, method=method)
  
  # z_traj: [Steps, B, H_pred, h, w, d]
  # returen t=0 and full trajectory
  return z_traj[-1], z_traj


def visualize_trajectory_rae(gt_images, gt_rae_decoded, pred_rae_decoded_list):
    def to_numpy_images(x):
        if torch.is_tensor(x):
            x = x.detach().float().cpu().numpy()
        # [T, 3, H, W] -> [T, H, W, 3]
        if x.ndim == 4 and x.shape[1] == 3:
            x = np.transpose(x, (0, 2, 3, 1))
        return np.clip(x, 0, 1)

    gt_images = to_numpy_images(gt_images)
    gt_rae_decoded = to_numpy_images(gt_rae_decoded)
    pred_rae_decoded_list = [to_numpy_images(p) for p in pred_rae_decoded_list]

    num_samples = len(pred_rae_decoded_list)
    H_len = gt_images.shape[0]

    # Rows: RGB + GT RAE + each sample RAE decode
    rows = 2 + num_samples
    cols = H_len
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), constrained_layout=True)
    if cols == 1:
        axes = axes[:, np.newaxis]
    if rows == 1:
        axes = axes[np.newaxis, :]

    for t in range(H_len):
        ax = axes[0, t]
        ax.imshow(gt_images[t])
        ax.axis('off')
        if t == 0:
            ax.set_title("Ground Truth (RGB)", fontweight="bold")

    for t in range(H_len):
        ax = axes[1, t]
        ax.imshow(gt_rae_decoded[t])
        ax.axis('off')
        if t == 0:
            ax.set_title("GT (RAE Decode)")

    for i in range(num_samples):
        for t in range(H_len):
            ax = axes[i + 2, t]
            ax.imshow(pred_rae_decoded_list[i][t])
            ax.axis('off')
            if t == 0:
                ax.set_title(f"Sample {i + 1} (RAE Decode)")

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return image


@torch.inference_mode()
def plot_samples(model, batch, device, cfg: DictConfig):
    raw_model = model.module if hasattr(model, "module") else model
    model_dtype = next(raw_model.parameters()).dtype
    
    n_ctx = raw_model.context_frames
    H_pred = raw_model.prediction_frames
    inv_transform = make_inverse_transforms(img_size=224)
    rae_model = get_rae_decoder(cfg, device)
    remove_dino_affine_for_rae = bool(cfg.vae.get("remove_dino_norm_affine_for_rae", True))

    # 1. Prepare Data
    if isinstance(batch, (list, tuple)):
        if isinstance(batch[0], dict):
            x = batch[0]['visual'].to(device=device, dtype=model_dtype, non_blocking=True)
            proprio = batch[0]['proprio'].to(device=device, dtype=model_dtype, non_blocking=True)
        else:
            x = batch[0].to(device=device, dtype=model_dtype, non_blocking=True)
            proprio = None
        action = batch[1].to(device=device, dtype=model_dtype, non_blocking=True) if len(batch) > 1 else None
    else:
        x = batch.to(device=device, dtype=model_dtype, non_blocking=True)
        action, proprio = None, None

    with torch.amp.autocast('cuda', enabled=True, dtype=model_dtype):
        # =========================================================
        # Step 1: Extract DINO Features (GT)
        # =========================================================
        # preprocess handles: RGB -> DINO Extract -> Normalization
        # Output: [B, T, H, W, 3072]
        dino_features = raw_model.vae.preprocess(x)
        
        # GT future DINO features for visualization
        gt_dino_visual = dino_features[0, n_ctx:n_ctx + H_pred, ...]

        # =========================================================
        # Step 2: Encode to Latent (for Prediction)
        # =========================================================
        # Flatten [B, T] dimensions for VAE input [B_total, H, W, C]
        dino_flat = einops.rearrange(dino_features, "b t h w c -> (b t) h w c")
        
        # Access inner VAE
        inner_vae = raw_model.vae.vae if hasattr(raw_model.vae, "vae") else raw_model.vae
        
        # Encode: DINO Features (3072) -> Latent (16)
        dist = inner_vae.encode(dino_flat)
        z_latent_flat = dist.sample() 
        
        # Reshape back to [B, T, H, W, 16]
        z_latent = einops.rearrange(z_latent_flat, "(b t) h w c -> b t h w c", b=dino_features.shape[0])
        
        # Prepare Context
        z_ctx_0 = z_latent[0:1, :n_ctx, ...]
        
        # =========================================================
        # Step 3: Prediction Loop
        # =========================================================
        num_viz_samples = 1 if getattr(raw_model, "is_regression_model", False) else 3
        batch_0_preds_latent = []
        
        for k in range(num_viz_samples):
            # Predict next latent steps
            z_pred_last, _ = sample(
                raw_model, 
                z_ctx_0, 
                action[0:1] if action is not None else None, 
                proprio[0:1] if proprio is not None else None, 
                steps=20
            )
            batch_0_preds_latent.append(z_pred_last[0])
        
        # =========================================================
        # Step 4: Decode Predictions (Latent -> DINO)
        # =========================================================
        preds_dino_visual = []
        for p in batch_0_preds_latent:
            # Input: [H_pred, h, w, 16] -> Decode -> [H_pred, h, w, 3072]
            decoded = inner_vae.decode(p)
            preds_dino_visual.append(decoded)

        # Prepare RGB ground truth for display
        gt_raw_frames = x[0, n_ctx:n_ctx + H_pred, ...].float().cpu()
        gt_rgb_images = inv_transform(gt_raw_frames) 
        gt_rgb_images = torch.clamp(gt_rgb_images, 0, 1)

    # =========================================================
    # Step 5: Visualization (RAE decode on last 768 dims)
    # =========================================================
    amp_off = (
        torch.amp.autocast(device_type='cuda', enabled=False)
        if device.type == "cuda" else nullcontext()
    )
    with amp_off:
        gt_rae_input = raw_model.vae.convert_to_rae_space(
            gt_dino_visual.float(),
            target_dim=768,
            remove_dino_norm_affine=remove_dino_affine_for_rae
        )
        gt_rae_decoded = rae_model.decode(gt_rae_input.permute(0, 3, 1, 2).float())

        pred_rae_decoded = []
        for pred in preds_dino_visual:
            pred_rae_input = raw_model.vae.convert_to_rae_space(
                pred.float(),
                target_dim=768,
                remove_dino_norm_affine=remove_dino_affine_for_rae
            )
            pred_img = rae_model.decode(pred_rae_input.permute(0, 3, 1, 2).float())
            pred_rae_decoded.append(pred_img)

    pil_image = visualize_trajectory_rae(gt_rgb_images, gt_rae_decoded, pred_rae_decoded)
    
    return pil_image


def train(
    model: DDP,
    ema: PredictorModel,
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
  ema.eval() # EMA is always Eval

  optimizer.zero_grad()
  step = initial_step
  
  img_size = 224
  transform = make_transforms(img_size=img_size)

  # Data Initialization
  context_len = cfg.model.context_frames
  pred_len = cfg.model.prediction_frames

  # initializing Dataloader
  (
    dataset, val_dataset, traj_dataset, val_traj_dataset, 
    train_dataloader, val_dataloader, dist_sampler, viz_val_loader
  ) = init_data(
      data_paths=cfg.data.paths,
      batch_size=cfg.training.batch_size,          
      num_workers=8,           
      dataset_type="custom",
      frameskip=cfg.data.frameskip,
      num_hist=context_len, 
      num_pred=pred_len,
      transform=transform,
      pin_mem=True,        
      persistent_workers=True, 
      rank=rank,             
      world_size=world_size,
      n_rollout=None 
  )
  
  train_sampler = dist_sampler
  for epoch in range(initial_epoch, cfg.training.epochs):
    # DDP set epoch (crucial for shuffling)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    for batch_idx, batch in enumerate(train_dataloader):
      objective = normalize_objective(cfg.objective)
      if objective == "CFM":
        loss = training_step(
            model,
            batch,
            cfg.training.mixed_precision,
            device
        )
      elif is_regression_objective(objective):
        loss = training_step_regression(
            model,
            batch,
            cfg.training.mixed_precision,
            device
        )
      else:
        raise NotImplementedError(f"Unsupported objective: {cfg.objective}")

      # Gradient Accumulation
      loss = loss / cfg.training.accumulation_steps
      loss.backward()

      if cfg.training.accumulation_steps == 1:
        should_step = True
      else:
        should_step = ((step + 1) % cfg.training.accumulation_steps == 0)

      if should_step:
        if "clip_grad_norm" in cfg.training:
          torch.nn.utils.clip_grad_norm_(
              model.parameters(),
              cfg.training.clip_grad_norm
          )
        optimizer.step()
        update_ema(ema, model.module)
        scheduler.step()
        optimizer.zero_grad()

      # Logging
      if step % cfg.training.log_every_steps == 0:
        step_loss = loss.item() * cfg.training.accumulation_steps
        if rank == 0:
            logger.info(
                f"[epoch={epoch:04d}, step={step:08d}] loss={step_loss:.8f}"
            )
            if cfg.training.wandb and wandb.run is not None:
              current_lr = optimizer.param_groups[0]["lr"]
              wandb.log({"train_loss": step_loss,
                        "lr": current_lr}, step=step)

      step += 1

      # Intra-epoch Checkpointing
      if step % 5000 == 0 and rank == 0:
          save(checkpoint_path / f"step_{step:08d}.pt", 
               model.module, ema, optimizer, scheduler, epoch, step, cfg)
          logger.info(f"Checkpoint saved at step {step}")


      if step % 3000 == 0:
          # --- Evaluation 逻辑 ---
          model.eval()
          eval_loader = viz_val_loader if viz_val_loader is not None else val_dataloader
          
          # 执行评估 (所有 rank 都会参与计算)
          (mean_future_mse, min_of_k_mse, mean_of_k_mse) = evaluate(
              model.module,
              eval_loader,
              device,
              objective=cfg.objective
          )
          model.train() # 评估完切回训练模式
          
          if rank == 0:
            logger.info(f"[Step Evaluation] step={step} val_future_mse: {mean_future_mse.item():.6f}")
            if cfg.training.wandb and wandb.run is not None:
              wandb.log({
                  "val_mean_future_mse": mean_future_mse.item(),
                  "val_min_of_k_mse": min_of_k_mse.item(),
              }, step=step)

          # --- Visualization 逻辑 ---
          # 所有的 rank 都需要通过这个循环来保证 dataloader 状态同步（或者仅由 rank 0 操作）
          for batch_viz in eval_loader:
            break
          
          if rank == 0:
              samples = plot_samples(model, batch_viz, device, cfg)
              save_dir = samples_path / f"step_visuals"
              save_dir.mkdir(exist_ok=True, parents=True)
              samples.save((save_dir / f"step_{step}.jpg"))
              
              if cfg.training.wandb and wandb.run is not None:
                wandb.log({"val_samples": wandb.Image(samples)}, step=step)
          
          # 分布式同步，防止 rank 0 处理图片太慢导致其他卡提前跑下一个 batch 导致死锁
          dist.barrier()


    # === Epoch End Operations ===
    if rank == 0:
      save(
          checkpoint_path / f"current.pt",
          model.module, ema, optimizer, scheduler, epoch, step, cfg
      )

    dist.barrier()

    # Regular Checkpointing
    if epoch % cfg.training.ckpt_every_epochs == 0:
      if rank == 0:
        save(
            checkpoint_path / f"epoch={epoch:04d}.pt",
            model.module, ema, optimizer, scheduler, epoch, step, cfg
        )

    dist.barrier()

    # Evaluation
    if epoch % cfg.training.evaluate_every_epochs == 0:
      model.eval()
      eval_loader = viz_val_loader if viz_val_loader is not None else val_dataloader
      
      # use model.module for inference
      (mean_future_mse, min_of_k_mse, mean_of_k_mse) = evaluate(
          model.module,
          eval_loader,
          device,
          objective=cfg.objective
      )
      model.train()
      
      if rank == 0:
        logger.info(f"[epoch={epoch}] val_future_mse: {mean_future_mse.item():.6f}")
        logger.info(f"[epoch={epoch}] val_min_k_mse:  {min_of_k_mse.item():.6f}")
        
        if cfg.training.wandb and wandb.run is not None:
          wandb.log({
              "val_mean_future_mse": mean_future_mse.item(),
              "val_min_of_k_mse": min_of_k_mse.item(),
              "val_mean_of_k_mse": mean_of_k_mse.item()
          }, step=step)

    dist.barrier()

    # Visualization
    if epoch % cfg.training.sample_every_epochs == 0:
      eval_loader = viz_val_loader if viz_val_loader is not None else val_dataloader
      # Get one batch
      for batch in eval_loader:
        break
      
      # Only Rank 0 visualizes
      if rank == 0:
          samples = plot_samples(model, batch, device, cfg)
          
          save_dir = samples_path / f"epoch_{epoch}"
          save_dir.mkdir(exist_ok=True, parents=True)
          samples.save((save_dir / f"step_{step}.jpg"))
          
          if cfg.training.wandb and wandb.run is not None:
            wandb.log({"val_samples": wandb.Image(samples)}, step=step)

    dist.barrier()


config_path = os.environ.get(
    "CONFIG_PATH",
    "/home/lizijia/Gen-JEPA/vae/models/configs/kubric"
)
config_name = os.environ.get(
    "CONFIG_NAME",
    "dino_foresight_cfm.yaml"
)


@hydra.main(
    version_base=None,
    config_path=config_path,
    config_name=config_name
)
def main(cfg: DictConfig):
  # Setup DDP
  rank, world_size, local_rank, device = setup_ddp()
  seed_everything(cfg.training.seed + rank)

  if rank == 0:
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

  # Directories
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


  # --- Model Initialization ---
  # clear GPU memory before model initialization to reduce fragmentation and OOM risk
  torch.cuda.empty_cache()

  objective = normalize_objective(cfg.objective)

  if "vae" in cfg:
    if objective == "CFM":
      model = AutoregressiveDINOForesightLatent(
          **cfg.model,
          vae=cfg.vae
      )
    elif is_regression_objective(objective):
      model = AutoregressiveDINOForesightLatentRegressor(
          **cfg.model,
          vae=cfg.vae
      )
    else:
      raise NotImplementedError(f"Unsupported objective: {cfg.objective}")
  else:
     raise NotImplementedError("Only AutoregressiveDINOForesightLatent with VAE is implemented.")
    
  # clone EMA on CPU to save GPU memory
  ema = deepcopy(model)
  for p in ema.parameters():
    p.requires_grad = False
  
  # move to device and convert to bfloat16 for training
  model = model.to(device).bfloat16()
  ema = ema.to(device).bfloat16()

  # wrap with DDP
  model = DDP(
      model, 
      device_ids=[local_rank], 
      output_device=local_rank,
      find_unused_parameters=False 
  )

  if rank == 0:
      try:
          summary(model.module.transformer, depth=3)
      except Exception as e:
          logger.warning(f"Could not print summary: {e}")

  # --- Resume Logic ---
  if "resume" in cfg:
    # map_location='cpu' to prevent GPU OOM
    ckpt = torch.load(
        cfg.resume.ckpt,
        map_location="cpu",  # Load to CPU first
        weights_only=True
    )
    # loading weights to DDP module
    model.module.load_state_dict(ckpt["model_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    if rank == 0:
        logger.info(f"restored ckpt: {cfg.resume.ckpt}")
    initial_epoch = ckpt["epoch"] + 1
    initial_step = ckpt["step"] + 1
    
    # clean up
    del ckpt
    torch.cuda.empty_cache()
    
  # --- WandB (Rank 0 Only) ---
  if cfg.training.wandb and rank == 0:
    run_id = None
    # reload ckpt meta (lightweight) to get wandb run id for resuming
    if "resume" in cfg and cfg.resume.wandb:
        ckpt_meta = torch.load(cfg.resume.ckpt, map_location="cpu", weights_only=True)
        run_id = ckpt_meta.get("wandb", {}).get("run", {}).get("id", None)
        del ckpt_meta

    wandb.init(
        project="gen-jepa_vae",
        name="gen-jepa_kubric-vae_rae",
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow" if run_id else None,
        id=run_id
    )

  # Optimizer
  optimizer = AdamW(model.parameters(), lr=cfg.training.lr)

  # Scheduler
  match cfg.training.lr_scheduler.name:
    case "cosine":
      scheduler = CosineAnnealingLR(
          optimizer=optimizer,
          T_max=cfg.training.steps
      )
    case "linear_warmup":
      scheduler = LinearWarmupScheduler(
          optimizer=optimizer,
          warmup_steps=cfg.training.lr_scheduler.warmup_steps,
          lr_max=cfg.training.lr
      )
    case _: raise NotImplementedError()

  # --- Resume Optimizer & Scheduler ---
  # since we del ckpt above to save memory, we need to reload 
  # it here for optimizer/scheduler state restoration
  if "resume" in cfg:
      if rank == 0:
          logger.info(f"Reloading checkpoint for optimizer/scheduler state: {cfg.resume.ckpt}")
      
      # Map to CPU strictly to avoid GPU OOM
      ckpt = torch.load(
          cfg.resume.ckpt, 
          map_location="cpu", 
          weights_only=True
      )
      
      # recover Optimizer
      if "optimizer_state_dict" in ckpt:
          try:
              optimizer.load_state_dict(ckpt["optimizer_state_dict"])
          except Exception as e:
              if rank == 0: 
                  logger.warning(f"Failed to load optimizer state: {e}")
      
      # recover Scheduler
      if "scheduler_state_dict" in ckpt:
          scheduler.load_state_dict(ckpt["scheduler_state_dict"])
      
      # clean up
      del ckpt
      torch.cuda.empty_cache()

  # --- Start Training ---
  train(
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


if __name__ == '__main__':
  main()
