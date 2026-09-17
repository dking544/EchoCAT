from __future__ import annotations

import cv2
import numpy as np


def _gray_image(image):
    if image.ndim == 2:
        return image
    return image[..., 0]


def _three_channel_uint8(gray):
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


class UltrasoundStyleAug:


    def __init__(self, p=0.8):
        self.p = float(p)

    def __call__(self, results):
        if np.random.random() >= self.p:
            return results
        for key in results.get("img_fields", ["img"]):
            gray = _gray_image(results[key]).astype(np.float32) / 255.0

            gamma = np.random.uniform(0.70, 1.45)
            contrast = np.random.uniform(0.80, 1.20)
            brightness = np.random.uniform(-0.06, 0.06)
            gray = np.power(np.clip(gray, 0.0, 1.0), gamma)
            gray = (gray - gray.mean()) * contrast + gray.mean() + brightness

            speckle_sigma = np.random.uniform(0.0, 0.08)
            additive_sigma = np.random.uniform(0.0, 0.018)
            if speckle_sigma > 0:
                gray = gray + gray * np.random.normal(
                    0.0, speckle_sigma, gray.shape).astype(np.float32)
            if additive_sigma > 0:
                gray = gray + np.random.normal(
                    0.0, additive_sigma, gray.shape).astype(np.float32)

            if np.random.random() < 0.35:
                sigma = np.random.uniform(0.25, 1.20)
                gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma,
                                        sigmaY=sigma)
            elif np.random.random() < 0.35:
                sigma = np.random.uniform(0.35, 1.0)
                blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma,
                                           sigmaY=sigma)
                gray = gray + np.random.uniform(0.15, 0.45) * (gray - blurred)

            results[key] = _three_channel_uint8(gray * 255.0)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class FourierAmplitudeMixWithCache:


    def __init__(self, p=0.5, beta_min=0.02, beta_max=0.10,
                 cache_size=4):
        self.p = float(p)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.cache_size = int(cache_size)
        self._cache = {0: [], 1: []}

    def __call__(self, results):
        label_value = results.get("gt_label", -1)
        try:
            label = int(np.asarray(label_value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            label = -1

        for key in results.get("img_fields", ["img"]):
            gray = _gray_image(results[key]).astype(np.float32)
            donors = self._cache.setdefault(label, [])
            output = gray
            if donors and np.random.random() < self.p:
                donor = donors[np.random.randint(len(donors))]
                if donor.shape != gray.shape:
                    donor = cv2.resize(
                        donor, (gray.shape[1], gray.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
                current_fft = np.fft.fftshift(np.fft.fft2(gray))
                donor_fft = np.fft.fftshift(np.fft.fft2(donor))
                current_amp = np.abs(current_fft)
                donor_amp = np.abs(donor_fft)
                phase = np.angle(current_fft)

                height, width = gray.shape
                beta = np.random.uniform(self.beta_min, self.beta_max)
                half_h = max(1, int(round(height * beta / 2.0)))
                half_w = max(1, int(round(width * beta / 2.0)))
                center_h, center_w = height // 2, width // 2
                hs = slice(max(0, center_h - half_h),
                           min(height, center_h + half_h + 1))
                ws = slice(max(0, center_w - half_w),
                           min(width, center_w + half_w + 1))
                lam = np.random.uniform(0.30, 0.70)
                mixed_amp = current_amp.copy()
                mixed_amp[hs, ws] = (
                    lam * current_amp[hs, ws]
                    + (1.0 - lam) * donor_amp[hs, ws]
                )
                mixed_fft = mixed_amp * np.exp(1j * phase)
                output = np.real(np.fft.ifft2(np.fft.ifftshift(mixed_fft)))

            donors.append(gray.copy())
            if len(donors) > self.cache_size:
                del donors[0]
            results[key] = _three_channel_uint8(output)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(p={self.p}, "
                f"beta=[{self.beta_min}, {self.beta_max}])")


class RandomConvolutionStyle:


    def __init__(self, p=0.5, max_kernel_size=5):
        self.p = float(p)
        self.max_kernel_size = int(max_kernel_size)

    def __call__(self, results):
        if np.random.random() >= self.p:
            return results
        valid_sizes = [size for size in (1, 3, 5, 7)
                       if size <= self.max_kernel_size]
        kernel_size = int(np.random.choice(valid_sizes))
        kernel = np.random.normal(
            0.0, 1.0, (kernel_size, kernel_size)).astype(np.float32)
        kernel /= max(float(np.sqrt((kernel ** 2).sum())), 1e-6)

        for key in results.get("img_fields", ["img"]):
            gray = _gray_image(results[key]).astype(np.float32)
            filtered = cv2.filter2D(
                gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
            filtered = (filtered - filtered.mean()) / (filtered.std() + 1e-6)
            filtered = filtered * (gray.std() + 1e-6) + gray.mean()
            strength = np.random.uniform(0.20, 0.55)
            output = (1.0 - strength) * gray + strength * filtered
            results[key] = _three_channel_uint8(output)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(p={self.p}, "
                f"max_kernel_size={self.max_kernel_size})")
