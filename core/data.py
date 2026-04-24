import os
import json
from pathlib import Path
from typing import List
from typing import Optional, Callable, Union, Tuple, Iterator

import cv2
import numpy as np
import torch
from core.transforms import get_transforms
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Normalize

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']


def _swap_path_segment(path: Path, *, src: str, dst: str) -> Path:
    parts = list(path.parts)
    idx = None
    for i, part in enumerate(parts):
        if part.lower() == src.lower():
            idx = i
    if idx is not None:
        parts[idx] = dst
        return Path(*parts)
    return path


def image_path_to_annotation_path(image_path: str, label_format: str) -> str:
    p = Path(image_path)
    if label_format == "npy":
        return str(p.with_suffix(".npy"))
    if label_format == "json":
        ann = _swap_path_segment(p, src="images", dst="annotations")
        return str(ann.with_suffix(".json"))
    raise ValueError(f"Unsupported label_format: {label_format}")


def _infer_label_format(image_paths: List[str]) -> str:
    """
    Decide dataset label format once and enforce it consistently.

    Rules:
      - If both JSON and NPY exist for a sample -> error.
      - If neither exists for all samples -> error.
      - Otherwise choose the first discovered format and require it for all.
    """
    chosen: Optional[str] = None
    for image_path in image_paths:
        npy_path = image_path_to_annotation_path(image_path, "npy")
        json_path = image_path_to_annotation_path(image_path, "json")
        has_npy = os.path.isfile(npy_path)
        has_json = os.path.isfile(json_path)
        if has_npy and has_json:
            raise ValueError(f"Both .npy and .json annotations exist for {image_path}")
        if has_npy or has_json:
            chosen = "npy" if has_npy else "json"
            break

    if chosen is None:
        raise FileNotFoundError("No annotation files found (expected either .npy or canonical .json).")

    for image_path in image_paths:
        ann_path = image_path_to_annotation_path(image_path, chosen)
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"Missing {chosen} annotation for {image_path}: {ann_path}")

    return chosen


def load_points_from_json(annotation_path: str) -> np.ndarray:
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "points" not in payload:
        raise ValueError(f"Invalid canonical annotation JSON (missing 'points'): {annotation_path}")
    points = payload["points"]
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.array(points, dtype=np.float32).reshape(-1, 2)
    return arr


def generate_density_map(
    label: Tensor,
    height: int,
    width: int,
    sigma: Optional[float] = None,
) -> Tensor:
    """
    Generate the density map based on the dot annotations provided by the label.
    """
    density_map = torch.zeros((1, height, width), dtype=torch.float32)

    if len(label) > 0:
        assert len(label.shape) == 2 and label.shape[1] == 2, f"label should be a Nx2 tensor, got {label.shape}."
        label_ = label.long()
        label_[:, 0] = label_[:, 0].clamp(min=0, max=width - 1)
        label_[:, 1] = label_[:, 1].clamp(min=0, max=height - 1)
        density_map[0, label_[:, 1], label_[:, 0]] = 1.0

    if sigma is not None:
        assert sigma > 0, f"sigma should be positive if not None, got {sigma}."
        density_map = torch.from_numpy(gaussian_filter(density_map, sigma=sigma))

    return density_map


def collate_fn(
    batch: List[Tensor]
) -> Tuple[Tensor, List[Tensor], Optional[Tensor], List[str], List[np.ndarray]]:
    batch = list(zip(*batch))
    images = batch[0]
    assert len(images[0].shape) == 4, f"images should be a 4D tensor, got {images[0].shape}."
    images = torch.cat(images, 0)
    points = batch[1]  # list of lists of tensors, flatten it
    points = [p for points_ in points for p in points_]

    # densities can be disabled for point-based models
    if all(d is None for d in batch[2]):
        densities = None
    else:
        densities = torch.cat(batch[2], 0)
    data_paths = batch[3]  # list of lists of strings, flatten it
    data_paths = [path for path_ in data_paths for path in path_]
    original_images = batch[4]
    original_images = [img for img_ in original_images for img in img_]
    return images, points, densities, data_paths, original_images

class DatasetWithLabels(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        input_size: int,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        num_crops: int = 1,
        need_density: bool = True,
        return_meta: bool = True,
    ) -> None:
        self.image_paths = self.get_image_paths(dataset_path)
        self.split = split
        self.label_format = _infer_label_format(self.image_paths)

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms = transforms

        self.sigma = sigma
        self.num_crops = num_crops
        self.input_size = input_size
        self.need_density = need_density
        self.return_meta = return_meta

    def get_image_paths(self, dataset_path: str) -> List[str]:
        with open(dataset_path, "r") as f:
            image_paths = f.read().splitlines()
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, List[Tensor], Optional[Tensor], List[str], List[np.ndarray]]:
        image_path = self.image_paths[idx]
        label_path = image_path_to_annotation_path(image_path, self.label_format)

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        original_image = image.copy() if self.return_meta else None
        image = self.to_tensor(image)

        if self.label_format == "npy":
            with open(label_path, "rb") as f:
                label = np.load(f)
        else:
            label = load_points_from_json(label_path)

        label = torch.from_numpy(label).float()
        if self.transforms is not None:
            images_labels = [self.transforms(image.clone(), label.clone()) for _ in range(self.num_crops)]
            images, labels = zip(*images_labels)
        else:
            images = [image.clone() for _ in range(self.num_crops)]
            labels = [label.clone() for _ in range(self.num_crops)]

        images = [self.normalize(img) for img in images]

        if self.need_density:
            density_maps = torch.stack(
                [generate_density_map(label, image.shape[-2], image.shape[-1], sigma=self.sigma) for image, label in zip(images, labels)],
                0,
            )
        else:
            density_maps = None

        if self.return_meta:
            data_paths = [image_path] * len(images)
            original_images = [original_image] * len(images)  # type: ignore[list-item]
        else:
            data_paths = []
            original_images = []
        images = torch.stack(images, 0)
        return images, labels, density_maps, data_paths, original_images


class DatasetWithoutLabels(Dataset):
    def __init__(
        self,
        dataset_path: str,
        input_size: int
    ) -> None:
        self.image_paths, self.video_paths = self.get_media_paths(dataset_path)
        self.data_paths = self.image_paths + self.video_paths
        ni, nv = len(self.image_paths), len(self.video_paths)

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.input_size = input_size
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(self.video_paths):
            self.new_video(self.video_paths[0])
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found.'

    def get_media_paths(self, dataset_path: str) -> Tuple[List[str], List[str]]:
        image_paths, video_paths = [], []
        for root, _, files in os.walk(dataset_path):  # Walk through the folder
            for file in files:
                file_path = os.path.join(root, file)
                if file.split(".")[-1].lower() in img_formats:
                    image_paths.append(file_path)
                elif file.split(".")[-1].lower() in vid_formats:
                    video_paths.append(file_path)
        return image_paths, video_paths

    def __len__(self) -> int:
        return self.nf

    def __iter__(self) -> Iterator:
        self.count = 0
        return self

    def __next__(self) -> Tuple[Tensor, np.ndarray, str]:
        if self.count == self.nf:
            raise StopIteration
        data_path = self.data_paths[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret, original_image = self.cap.read()
            if not ret:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    data_path = self.data_paths[self.count]
                    self.new_video(data_path)
                    ret, original_image = self.cap.read()

            self.frame += 1

        else:
            # Read image
            self.count += 1
            original_image = cv2.imread(data_path)  # BGR
            assert original_image is not None, 'Image Not Found ' + data_path

        # image = cv2.resize(original_image, (self.input_size, self.input_size))
        image = self.to_tensor(original_image)
        image = self.normalize(image)

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        return image, original_image, data_path

    def new_video(self, video_path: str) -> None:
        self.frame = 0
        self.cap = cv2.VideoCapture(video_path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_dataloader(
    config: object,
    split: str = "train",
    ddp: bool = False,
) -> Union[Tuple[DataLoader, Union[DistributedSampler, None]], DataLoader]:
    if split == "train":  # train, strong augmentation
        transforms = get_transforms(config)
    else:
        transforms = None

    split = "valid" if split == "val" else split

    # Point-based vs density-based models (avoid generating density maps when not needed)
    density_based_networks = {"clip_ebc", "dmcount", "fusioncount", "ffnet", "steerer"}
    need_density = getattr(config, "network", None) in density_based_networks

    # Train/val do not need paths/original images (saves CPU+RAM), test keeps them for visualization.
    return_meta = split == "test"

    dataset = DatasetWithLabels(
        dataset_path=os.path.join("./datasets", f"{split}.txt"),
        split=split,
        input_size=config.input_size,
        transforms=transforms,
        sigma=None,
        num_crops=config.num_crops if split == "train" else 1,
        need_density=need_density,
        return_meta=return_meta,
    )

    num_workers = int(getattr(config, "num_workers", 0))
    base_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if num_workers > 0 and split == "train":
        # Keep workers alive between epochs and prefetch a small number of batches.
        base_loader_kwargs["persistent_workers"] = True
        base_loader_kwargs["prefetch_factor"] = 2
    if ddp and split == "train":  # data_loader for training in DDP
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            **base_loader_kwargs,
        )
        return data_loader, sampler

    elif split == "train":  # data_loader for training
        data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            **base_loader_kwargs,
        )
        return data_loader, None

    else:  # data_loader for evaluation
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Use batch size 1 for evaluation
            shuffle=False,
            **base_loader_kwargs,
        )
        return data_loader

