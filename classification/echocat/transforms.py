import numpy as np
import cv2
import torch
from echocat.style import UltrasoundStyleAug

class BGRToGrayThreeChannel:


    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            image = results[key]
            if image.ndim == 2:
                gray = image
            elif image.shape[2] == 1:
                gray = image[..., 0]
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            results[key] = np.repeat(gray[..., None], 3, axis=2)
        return results

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ForegroundPercentileWindowThreeChannel:


    def __init__(self, lower=1.0, upper=99.0, background_max=5):
        self.lower = float(lower)
        self.upper = float(upper)
        self.background_max = int(background_max)

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            image = results[key]
            if image.ndim == 2:
                gray = image
            elif image.shape[2] == 1:
                gray = image[..., 0]
            else:

                gray = image[..., 0]
            foreground = gray > self.background_max
            if not np.any(foreground):
                output = gray.astype(np.uint8, copy=True)
            else:
                values = gray[foreground].astype(np.float32)
                low, high = np.percentile(values, (self.lower, self.upper))
                if high <= low + 1e-6:
                    output = gray.astype(np.uint8, copy=True)
                else:
                    mapped = (
                        (gray.astype(np.float32) - low)
                        * (255.0 / (high - low))
                    )
                    output = np.rint(mapped).clip(0, 255).astype(np.uint8)
                    output[~foreground] = 0
            results[key] = np.repeat(output[..., None], 3, axis=2)
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(lower={self.lower}, "
            f"upper={self.upper}, background_max={self.background_max})"
        )


class PairedUltrasoundStyleToTensor:


    def __init__(self, p=0.8, mean=None, std=None):
        self.p = float(p)
        self.mean = np.asarray(
            mean or [123.675, 116.28, 103.53], dtype=np.float32,
        ).reshape(1, 1, 3)
        self.std = np.asarray(
            std or [58.395, 57.12, 57.375], dtype=np.float32,
        ).reshape(1, 1, 3)
        self.style = UltrasoundStyleAug(p=self.p)

    def _to_tensor(self, image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {image.shape}")


        normalized = (image[..., ::-1].astype(np.float32) - self.mean) / self.std
        chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
        return torch.from_numpy(chw)

    def __call__(self, results):
        keys = list(results.get("img_fields", ["img"]))
        if keys != ["img"]:
            raise ValueError(
                "PairedUltrasoundStyleToTensor expects only the primary img field"
            )
        clean = np.asarray(results["img"]).copy()
        styled_results = {"img": clean.copy(), "img_fields": ["img"]}
        styled = self.style(styled_results)["img"]
        results["img"] = torch.stack(
            (self._to_tensor(clean), self._to_tensor(styled)), dim=0,
        )
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
