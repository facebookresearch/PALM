from systems.criterions import PSNR, SSIM
from common.xdict import xdict

criterions = {
    "psnr": PSNR(),
    "ssim": SSIM(),
    # 'lpips': LPIPS(),
    # 'normal_error': NormalError()
}


def compute_metrics(out, batch):
    metrics = {}
    W = int(batch["width"])
    H = int(batch["height"])

    rf_psnr = criterions["psnr"](
        out["comp_rgb_full"].to(batch["rgb"]),
        batch["rgb"],
        valid_mask=batch["valid_mask"].view(-1) if "valid_mask" in batch else None,
    )
    rf_ssim = criterions["ssim"](
        out["comp_rgb_full"].to(batch["rgb"]).reshape(H, W, 3),
        batch["rgb"].reshape(H, W, 3),
        valid_mask=batch["valid_mask"].reshape(H, W) if "valid_mask" in batch else None,
    )
    metrics["rf_psnr"] = rf_psnr
    metrics["rf_ssim"] = rf_ssim

    if "comp_rgb_phys_full" in out:
        pbr_psnr = criterions["psnr"](
            out["comp_rgb_phys_full"].to(batch["rgb"]),
            batch["rgb"],
            valid_mask=batch["valid_mask"].view(-1) if "valid_mask" in batch else None,
        )
        pbr_ssim = criterions["ssim"](
            out["comp_rgb_phys_full"].to(batch["rgb"]).reshape(H, W, 3),
            batch["rgb"].reshape(H, W, 3),
            valid_mask=(
                batch["valid_mask"].reshape(H, W) if "valid_mask" in batch else None
            ),
        )

        metrics["pbr_psnr"] = pbr_psnr
        metrics["pbr_ssim"] = pbr_ssim

    metrics = xdict(metrics).prefix("metrics/")
    return metrics
