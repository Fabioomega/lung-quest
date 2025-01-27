import pretrainedmodels
import torch
import torchvision
import base64
import cv2

import pandas as pd
import numpy as np
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import risk_assessor.SimpleArchs as SimpleArchs

from fastai.vision.learner import create_head
from itertools import islice
from typing import List


def save_to_file(s: str, filename: str):
    with open(filename, "w") as file:
        file.write(s)


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


def load_model(
    model_arch: str, out_nodes: int, weights_path: str | None = None
) -> torch.nn.Module:

    model = None

    if model_arch == "inceptionv4":
        arch = pretrainedmodels.__dict__["inceptionv4"](
            num_classes=1000, pretrained="imagenet"
        )

        custom_head = create_head(nf=768 * 2, n_out=out_nodes)

        model = nn.Sequential(nn.Sequential(*list(arch.children())[:-2]), custom_head)

    elif model_arch == "resnet34":

        custom_head = create_head(nf=512, n_out=out_nodes)
        model = nn.Sequential(
            nn.Sequential(*list(torchvision.models.resnet34().children())[:-2]),
            custom_head,
        )

    elif model_arch == "tiny":
        model = SimpleArchs.get_simple_model("Tiny", out_nodes)
    else:
        print(
            "Architecture type: " + model_arch + " not supported. "
            "Please, make sure the `model_spec` CSV is found in the working directory and can be accessed."
        )
        return None

    if not weights_path is None:
        model.load_state_dict(torch.load(weights_path, weights_only=True))

    model.eval()
    return model


class XRayMortalityAssessor:

    def __init__(
        self,
        device: torch.device | str = torch.get_default_device(),
        assessor_folder: str = "risk_assessor",
        batch_size: int = 2,
    ):

        model_details_fn = f"{assessor_folder}/model_data/CXR_Lung_Risk_Specs.csv"
        ensemble_weights_fn = f"{assessor_folder}/model_data/ensemble_weights.csv"
        model_weights_path = f"{assessor_folder}/model_weights/ensamble-model-weights/"

        model_details_df = pd.read_csv(model_details_fn)
        ensemble_weights_df = pd.read_csv(ensemble_weights_fn)
        model_name = "Lung_Age_081221"

        self.n_models = model_details_df.shape[0]
        self.device = device
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToDtype(torch.float32, True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # parse the LASSO ensemble weights from the CSV file shared with the repository
        self.ensemble_weights = (
            torch.from_numpy(ensemble_weights_df["weight"].values)
            .float()
            .to(self.device)
        )

        # define the LASSO ensemble intercept computed on the tuning set
        self.lasso_intercept = 49.8484258

        self.models = []

        for model_id in range(self.n_models):
            out_nodes = int(model_details_df.Num_Classes[model_id])
            model_arch = model_details_df.Architecture[model_id].lower()
            path = model_weights_path + model_name + "_" + str(model_id) + ".pth"

            self.models.append(load_model(model_arch, out_nodes, path).to(self.device))

    def _preprocess(self, batched_imgs: np.ndarray) -> torch.TensorType:
        return self.transform(torch.from_numpy(batched_imgs)).to(self.device)

    @torch.no_grad()
    def classify_by_path(self, imgs_path: List[str]) -> List[float]:
        output = []
        for paths in batched(imgs_path, self.batch_size):
            imgs = [fix_channels(cv2.imread(path)) for path in paths]
            batch = self._preprocess(np.stack(imgs))
            pred_arr = torch.zeros((len(paths), self.n_models)).to(self.device)
            for i, model in enumerate(self.models):
                pred_arr[:, i] = model(batch)

            # compute the final CXR-Lung-Risk by ensembling the scores
            predictions = (
                torch.matmul(pred_arr, self.ensemble_weights) + self.lasso_intercept
            )

            output.extend(predictions.tolist())
        return output

    @torch.no_grad()
    def batch_all_classify_by_b64(self, encoded_seq: List[str]) -> List[float]:
        imgs = [fix_channels(base64_to_img(encoded)) for encoded in encoded_seq]
        batch = self._preprocess(np.stack(imgs))
        pred_arr = torch.zeros((len(encoded_seq), self.n_models)).to(self.device)

        for i, model in enumerate(self.models):
            pred_arr[:, i] = model(batch)

        predictions = (
            torch.matmul(pred_arr, self.ensemble_weights) + self.lasso_intercept
        )
        return predictions.tolist()


if __name__ == "__main__":
    assessor = XRayMortalityAssessor("cuda")
    print(assessor.classify_by_path(["00003924_001.png"]))
