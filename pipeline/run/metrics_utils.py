from __future__ import annotations

import numpy as np

try:
        from skimage.metrics import peak_signal_noise_ratio as sk_psnr
        from skimage.metrics import structural_similarity as sk_ssim

        _HAS_SKIMAGE = True
except Exception:
        _HAS_SKIMAGE = False


def _to_float_img(img: np.ndarray) -> np.ndarray:
        arr = img.astype(np.float32) / 255.0
        return arr


def compute_psnr(img_ref: np.ndarray, img_cmp: np.ndarray) -> float:
        """Compute PSNR between two RGB images (uint8 numpy arrays)."""
        if img_ref.shape != img_cmp.shape:
                raise ValueError("Images must have same shape for PSNR")
        if _HAS_SKIMAGE:
                return float(sk_psnr(img_ref, img_cmp, data_range=255.0))

        ref = _to_float_img(img_ref)
        cmp = _to_float_img(img_cmp)
        mse = float(np.mean((ref - cmp) ** 2))
        if mse == 0:
                return float("inf")
        return 20.0 * float(np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(
        img_ref: np.ndarray, img_cmp: np.ndarray, multichannel: bool = False
) -> float:
        """Compute SSIM between two RGB images. Returns value in [0,1].

        By default computes SSIM on luminance channel if skimage available and multichannel=False.
        """
        if img_ref.shape != img_cmp.shape:
                raise ValueError("Images must have same shape for SSIM")

        ref = _to_float_img(img_ref)
        cmp = _to_float_img(img_cmp)

        if ref.ndim == 3 and ref.shape[2] == 3:
                # Rec.601 luminance for default single-channel SSIM.
                ref_y = 0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]
                cmp_y = 0.299 * cmp[..., 0] + 0.587 * cmp[..., 1] + 0.114 * cmp[..., 2]
        else:
                ref_y = ref if ref.ndim == 2 else np.mean(ref, axis=-1)
                cmp_y = cmp if cmp.ndim == 2 else np.mean(cmp, axis=-1)

        if _HAS_SKIMAGE:
                if multichannel and ref.ndim == 3:
                        # skimage >= 0.19 uses channel_axis.
                        return float(
                                sk_ssim(
                                        ref,
                                        cmp,
                                        data_range=1.0,
                                        channel_axis=-1,
                                )
                        )
                return float(sk_ssim(ref_y, cmp_y, data_range=1.0))

        # fallback: compute SSIM on luminance channel using simple approximation
        # (ref_y/cmp_y already computed above)

        # compute simple SSIM (windowed) approximation using constants
        K1 = 0.01
        K2 = 0.03
        L = 1.0
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        mu1 = ref_y.mean()
        mu2 = cmp_y.mean()
        sigma1 = ref_y.var()
        sigma2 = cmp_y.var()
        cov = float(((ref_y - mu1) * (cmp_y - mu2)).mean())

        ssim = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / (
                (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 + sigma2 + C2)
        )
        return float(max(0.0, min(1.0, ssim)))
