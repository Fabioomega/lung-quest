import torchxrayvision as xrv
import torch, cv2, base64
import numpy as np
import torchvision.transforms.v2 as transforms

from torch import device, get_default_device
from typing import List, Dict
from functools import partial
from itertools import islice

ModelOutput = Dict[str, bool]


def batched(iterable, n):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def base64_to_img(encoded_data: str) -> cv2.Mat:
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def fix_channels(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img[np.newaxis, ...]

    return img[np.newaxis, :, :, 0]


def normalize(img: torch.Tensor, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception(
            "max image value ({}) higher than expected bound ({}).".format(
                img.max(), maxval
            )
        )

    img = (2 * (img.float() / maxval) - 1.0) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img


class XRayCenterCrop(object):
    """Perform a center crop on the long dimension of the input image"""

    def crop_center(self, img: np.ndarray) -> np.ndarray:
        _, _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty : starty + crop_size, startx : startx + crop_size]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.crop_center(img)


class XRayClassifier:

    def __init__(
        self,
        device: device | str = get_default_device(),
        densenet_weights: str = "densenet121-res224-all",
        batch_size: int = 2,
        threshold: float = 0.6,
    ):
        self.threshold = threshold
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [
                partial(normalize, maxval=255),
                XRayCenterCrop(),
                transforms.Resize(224),
            ]
        )

        self.model = (
            xrv.models.DenseNet(weights=densenet_weights).to(self.device).eval()
        )

        self.pathologies = self.model.pathologies

    def _convert_to_dict(self, predictions: List[List[float]]) -> ModelOutput:
        return [dict(zip(self.pathologies, prediction)) for prediction in predictions]

    def _preprocess(self, batched_imgs: np.ndarray) -> torch.TensorType:
        return self.transform(torch.from_numpy(batched_imgs)).to(self.device)

    @torch.no_grad()
    def classify_by_path(self, imgs_path: List[str]) -> ModelOutput:
        output = []
        for paths in batched(imgs_path, self.batch_size):
            imgs = [fix_channels(cv2.imread(path)) for path in paths]
            batch = self._preprocess(np.stack(imgs))
            out = self.model.forward(batch)
            output.extend(torch.where(out > 0.6, True, False).tolist())
        return self._convert_to_dict(output)

    @torch.no_grad()
    def raw_classify_by_path(self, imgs_path: List[str]):
        output = []
        for paths in batched(imgs_path, self.batch_size):
            imgs = [fix_channels(cv2.imread(path)) for path in paths]
            batch = self._preprocess(np.stack(imgs))
            out = self.model.forward(batch)
            output.extend((100 * out).tolist())
        return self._convert_to_dict(output)

    @torch.no_grad()
    def classify_by_b64(self, encoded_seq: List[str]) -> ModelOutput:
        output = []
        for encoded_batch in batched(encoded_seq, self.batch_size):
            imgs = [fix_channels(base64_to_img(encoded)) for encoded in encoded_batch]
            batch = self._preprocess(np.stack(imgs))
            out = self.model.forward(batch)
            output.extend(torch.where(out > 0.6, True, False).tolist())
        return self._convert_to_dict(output)

    @torch.no_grad()
    def batch_all_classify_by_b64(self, encoded_seq: List[str]) -> ModelOutput:
        imgs = [fix_channels(base64_to_img(encoded)) for encoded in encoded_seq]
        batch = self._preprocess(np.stack(imgs))
        out = self.model.forward(batch)
        return self._convert_to_dict(torch.where(out > 0.6, True, False).tolist())

    @torch.no_grad()
    def raw_batch_all_classify_by_b64(self, encoded_seq: List[str]) -> ModelOutput:
        imgs = [fix_channels(base64_to_img(encoded)) for encoded in encoded_seq]
        batch = self._preprocess(np.stack(imgs))
        out = self.model.forward(batch)
        return self._convert_to_dict(out.tolist())


if __name__ == "__main__":
    classifier = XRayClassifier("cuda")
    print(classifier.raw_classify_by_path(["teste.jpg"]))
