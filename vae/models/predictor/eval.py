import os
from contextlib import nullcontext
from pathlib import Path

import einops
import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from data_processing.transforms import make_inverse_transforms, make_transforms
from data_processing.utils import init_data
from vae.models.predictor.autoregressive_dino_foresight_latent import (
    AutoregressiveDINOForesightLatent,
    AutoregressiveDINOForesightLatentRegressor,
)
from vae.models.predictor.train import (
    evaluate,
    get_rae_decoder,
    is_regression_objective,
    normalize_objective,
    sample,
    visualize_trajectory_rae,
)
from vae.models.predictor.utilities.ddp import cleanup, setup_ddp
from vae.models.predictor.utilities.seed import seed_everything


config_path = os.environ.get(
    "CONFIG_PATH",
    "/home/lizijia/Gen-JEPA/vae/models/configs/kubric",
)
config_name = os.environ.get(
    "CONFIG_NAME",
    "dino_foresight_cfm.yaml",
)


def build_model(cfg: DictConfig):
    objective = normalize_objective(cfg.objective)
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)

    if "vae" not in cfg:
        raise NotImplementedError("Only VAE-based predictor is supported.")

    if objective == "CFM":
        model = AutoregressiveDINOForesightLatent(
            **model_kwargs,
            vae=cfg.vae,
        )
    elif is_regression_objective(objective):
        model = AutoregressiveDINOForesightLatentRegressor(
            **model_kwargs,
            vae=cfg.vae,
        )
    else:
        raise NotImplementedError(f"Unsupported objective: {cfg.objective}")
    return model


def load_checkpoint_to_model(model, ckpt_path: str, device: torch.device, rank: int):
    if not ckpt_path:
        raise ValueError("Must provide cfg.resume.ckpt.")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        if rank == 0:
            print("model_state_dict missing; fallback to ema_state_dict.")
        state_dict = ckpt["ema_state_dict"]

    fixed_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("module."):
            fixed_state_dict[k[7:]] = v
        else:
            fixed_state_dict[k] = v

    msg = model.load_state_dict(fixed_state_dict, strict=False)
    return ckpt, msg


def _build_perceptual_metrics(device: torch.device):
    try:
        import lpips  # type: ignore
    except ImportError as exc:
        raise ImportError("`lpips` is required for LPIPS metric.") from exc

    try:
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn  # type: ignore
    except ImportError:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn  # type: ignore

    lpips_metric = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_metric.parameters():
        p.requires_grad = False
    return lpips_metric, ssim_fn


def _unpack_batch(batch, device: torch.device, dtype: torch.dtype):
    if isinstance(batch, (list, tuple)):
        primary = batch[0]
        if isinstance(primary, dict):
            x = primary["visual"].to(device=device, dtype=dtype, non_blocking=True)
            proprio = primary.get("proprio")
            if proprio is not None:
                proprio = proprio.to(device=device, dtype=dtype, non_blocking=True)
        else:
            x = primary.to(device=device, dtype=dtype, non_blocking=True)
            proprio = None

        action = (
            batch[1].to(device=device, dtype=dtype, non_blocking=True)
            if len(batch) > 1 and torch.is_tensor(batch[1])
            else None
        )
        if len(batch) > 2 and proprio is None and torch.is_tensor(batch[2]):
            proprio = batch[2].to(device=device, dtype=dtype, non_blocking=True)
    else:
        x = batch.to(device=device, dtype=dtype, non_blocking=True)
        action, proprio = None, None
    return x, action, proprio


@torch.inference_mode()
def _compute_viz_sample_perceptual_metrics(model, batch, device: torch.device, cfg: DictConfig):
    lpips_metric, ssim_fn = _build_perceptual_metrics(device)
    rae_model = get_rae_decoder(cfg, device)
    raw_model = model.module if hasattr(model, "module") else model
    model_dtype = next(raw_model.parameters()).dtype
    data_dtype = torch.float32
    n_ctx = raw_model.context_frames
    h_pred = raw_model.prediction_frames
    objective = normalize_objective(cfg.objective)
    remove_affine = bool(cfg.vae.get("remove_dino_norm_affine_for_rae", True))
    inv_transform = make_inverse_transforms(img_size=224)

    x, action, proprio = _unpack_batch(batch, device, data_dtype)
    gt_raw_frames = x[0, n_ctx:n_ctx + h_pred, ...].float().cpu()
    gt_rgb_images = torch.clamp(inv_transform(gt_raw_frames), 0, 1)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=True, dtype=model_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast_ctx:
        # Keep this exactly aligned with train.py::plot_samples.
        dino_features = raw_model.vae.preprocess(x)
        gt_dino = dino_features[0, n_ctx:n_ctx + h_pred, ...]

        inner_vae = raw_model.vae.vae if hasattr(raw_model.vae, "vae") else raw_model.vae
        dino_flat = einops.rearrange(dino_features, "b t h w c -> (b t) h w c")
        z_latent_flat = inner_vae.encode(dino_flat).sample()
        z_latent = einops.rearrange(
            z_latent_flat,
            "(b t) h w c -> b t h w c",
            b=dino_features.shape[0],
        ).to(dtype=data_dtype)
        z_ctx_0 = z_latent[0:1, :n_ctx, ...]
        gt_latent = z_latent[0, n_ctx:n_ctx + h_pred, ...]

        num_viz_samples = 1 if getattr(raw_model, "is_regression_model", False) else 3
        pred_latent_list = []
        for _ in range(num_viz_samples):
            if is_regression_objective(objective) or getattr(raw_model, "is_regression_model", False):
                out = raw_model(
                    z_ctx_0.to(dtype=model_dtype),
                    action=action[0:1].to(dtype=model_dtype) if action is not None else None,
                    proprio=proprio[0:1].to(dtype=model_dtype) if proprio is not None else None,
                )
                z_pred = out[0] if isinstance(out, tuple) else out
            else:
                z_pred, _ = sample(
                    raw_model,
                    z_ctx_0,
                    action=action[0:1] if action is not None else None,
                    proprio=proprio[0:1] if proprio is not None else None,
                    steps=20,
                )
            pred_latent_list.append(z_pred[0].float())

    amp_off = (
        torch.amp.autocast(device_type="cuda", enabled=False)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_off:
        gt_rae = raw_model.vae.convert_to_rae_space(
            gt_dino.float(),
            target_dim=768,
            remove_dino_norm_affine=remove_affine,
        )
        gt_rae = einops.rearrange(gt_rae, "t h w c -> t c h w")
        gt_rgb = torch.clamp(rae_model.decode(gt_rae), 0, 1).float()

        sample_metrics = []
        pred_rgb_vis = []
        gt_latent_float = gt_latent.float()
        for pred_latent in pred_latent_list:
            pred_latent_clean = pred_latent.float()
            latent_mse = F.mse_loss(pred_latent_clean, gt_latent_float)

            pred_dino = inner_vae.decode(pred_latent_clean.to(dtype=model_dtype))
            pred_rae = raw_model.vae.convert_to_rae_space(
                pred_dino.float(),
                target_dim=768,
                remove_dino_norm_affine=remove_affine,
            )
            pred_rae = einops.rearrange(pred_rae, "t h w c -> t c h w")
            pred_rgb = torch.clamp(rae_model.decode(pred_rae), 0, 1).float()

            if pred_rgb.shape[-2:] != gt_rgb.shape[-2:]:
                pred_rgb = F.interpolate(pred_rgb, size=gt_rgb.shape[-2:], mode="bilinear", align_corners=False)

            try:
                ssim_val = ssim_fn(pred_rgb, gt_rgb, data_range=1.0, reduction="elementwise_mean")
            except TypeError:
                ssim_val = ssim_fn(pred_rgb, gt_rgb, data_range=1.0)
            if not torch.is_tensor(ssim_val):
                ssim_val = torch.tensor(float(ssim_val), device=device)
            if ssim_val.ndim > 0:
                ssim_val = ssim_val.mean()

            lpips_val = lpips_metric(pred_rgb * 2 - 1, gt_rgb * 2 - 1).view(-1).mean()
            sample_metrics.append(
                {
                    "latent_mse": float(latent_mse.item()),
                    "ssim": float(ssim_val.item()),
                    "lpips": float(lpips_val.item()),
                }
            )
            pred_rgb_vis.append(pred_rgb)

    if not sample_metrics:
        raise RuntimeError("No visualization samples were produced for SSIM/LPIPS.")

    mean_latent_mse = sum(v["latent_mse"] for v in sample_metrics) / len(sample_metrics)
    mean_ssim = sum(v["ssim"] for v in sample_metrics) / len(sample_metrics)
    mean_lpips = sum(v["lpips"] for v in sample_metrics) / len(sample_metrics)
    pil_image = visualize_trajectory_rae(gt_rgb_images, gt_rgb, pred_rgb_vis)
    rae_hw = (int(gt_rgb.shape[-2]), int(gt_rgb.shape[-1]))
    return pil_image, sample_metrics, mean_latent_mse, mean_ssim, mean_lpips, rae_hw


@hydra.main(
    version_base=None,
    config_path=config_path,
    config_name=config_name,
)
def main(cfg: DictConfig):
    rank, world_size, local_rank, device = setup_ddp()
    seed_everything(cfg.training.seed + rank)

    if rank == 0:
        print("--- Evaluation Mode (train.py-aligned) ---")
        print(f"Checkpoint: {cfg.resume.ckpt}")

    model = build_model(cfg).to(device).bfloat16().eval()
    ckpt, msg = load_checkpoint_to_model(model, cfg.resume.ckpt, device, rank)

    if rank == 0:
        print(f"Load Result: {msg}")

    transform = make_transforms(img_size=224)
    (
        _dataset,
        _val_dataset,
        _traj_dataset,
        _val_traj_dataset,
        _train_dataloader,
        val_dataloader,
        _dist_sampler,
        viz_val_loader,
    ) = init_data(
        data_paths=cfg.data.paths,
        batch_size=cfg.training.batch_size,
        num_workers=8,
        dataset_type="custom",
        frameskip=cfg.data.frameskip,
        num_hist=cfg.model.context_frames,
        num_pred=cfg.model.prediction_frames,
        transform=transform,
        pin_mem=True,
        persistent_workers=True,
        rank=rank,
        world_size=world_size,
        n_rollout=None,
    )

    eval_loader = viz_val_loader if viz_val_loader is not None else val_dataloader

    mean_future_mse, min_of_k_mse, mean_of_k_mse = evaluate(
        model,
        eval_loader,
        device,
        objective=cfg.objective,
    )

    for batch_viz in eval_loader:
        break
    else:
        raise RuntimeError("Validation dataloader is empty.")

    if rank == 0:
        step_num = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0
        viz_seed = int(cfg.training.seed) + int(step_num)
        torch.manual_seed(viz_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(viz_seed)
        pil_image, sample_metrics, mean_latent_mse, mean_ssim, mean_lpips, rae_hw = _compute_viz_sample_perceptual_metrics(
            model,
            batch_viz,
            device,
            cfg,
        )

        # Reset the same seed so the rendered samples are the same ones used for SSIM/LPIPS.
        torch.manual_seed(viz_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(viz_seed)

        print("\n[Evaluation Result]")
        print(f"Mean Future MSE: {mean_future_mse.item():.6f}")
        print(f"Min of K MSE:    {min_of_k_mse.item():.6f}")
        print(f"Mean of K MSE:   {mean_of_k_mse.item():.6f}")
        for i, m in enumerate(sample_metrics, start=1):
            print(f"Sample-{i} Latent MSE: {m['latent_mse']:.6f}")
            print(f"Sample-{i} SSIM:  {m['ssim']:.6f}")
            print(f"Sample-{i} LPIPS: {m['lpips']:.6f}")
        print(f"Mean Latent MSE: {mean_latent_mse:.6f}")
        print(f"Mean SSIM:       {mean_ssim:.6f}")
        print(f"Mean LPIPS:      {mean_lpips:.6f}")
        print(f"RAE decode size: {rae_hw[0]}x{rae_hw[1]}")
        print(f"Is 256x256:      {rae_hw == (256, 256)}")

        save_dir = Path(cfg.experiment_path) / cfg.name / "eval_results"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"eval_prediction_{step_num:08d}.jpg"
        pil_image.save(save_path)
        print(f"Saved visualization to: {save_path}")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
