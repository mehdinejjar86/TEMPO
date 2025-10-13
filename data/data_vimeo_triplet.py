import os, random
from typing import Tuple, List, Literal, Optional

import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

Mode = Literal["interp", "extrap_fwd", "extrap_bwd", "mix"]

class Vimeo90KTriplet(data.Dataset):
    """
    Vimeo-90K (triplet split) loader using list files:
      datasets/vimeo_triplet/tri_trainlist.txt
      datasets/vimeo_triplet/tri_testlist.txt

    Each line is "<scene>/<clip>", frames live at:
      datasets/vimeo_triplet/sequences/<scene>/<clip>/im1.png, im2.png, im3.png

    mode:
      - 'interp'     : (im1, im3) -> im2,  times [0,1], target 0.5
      - 'extrap_fwd' : (im1, im2) -> im3,  times [0,1], target 2.0
      - 'extrap_bwd' : (im2, im3) -> im1,  times [0,1], target -1.0
      - 'mix'        : sample one of the above each __getitem__
    """
    def __init__(
        self,
        root: str = "datasets/vimeo_triplet",
        split: Literal["train", "test"] = "train",
        mode: Mode = "interp",
        crop_size: Optional[int] = 192,
        aug_flip: bool = True,
        center_crop_eval: bool = True,
    ):
        super().__init__()
        self.root = root
        self.mode = mode
        self.crop_size = crop_size
        self.aug_flip = aug_flip
        self.center_crop_eval = center_crop_eval

        list_file = os.path.join(root, f"tri_{'train' if split=='train' else 'test'}list.txt")
        seq_root  = os.path.join(root, "sequences")

        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"Missing list file: {list_file}")
        if not os.path.isdir(seq_root):
            raise FileNotFoundError(f"Missing sequences directory: {seq_root}")

        with open(list_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        # Remove potential headers or comments
        self.items = [ln for ln in lines if not ln.startswith("#")]

        # Simple transforms
        self.to_tensor = T.ToTensor()  # keeps [0,1]

        # Decide if we’ll do center crop for eval
        self.is_train = (split == "train")

    def __len__(self) -> int:
        return len(self.items)

    def _load_triplet(self, rel_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load im1, im2, im3 → [3,H,W] each in [0,1]."""
        base = os.path.join(self.root, "sequences", rel_path)
        def load_png(name):
            p = os.path.join(base, f"{name}.png")
            with Image.open(p) as im:
                return self.to_tensor(im.convert("RGB"))
        im1 = load_png("im1")
        im2 = load_png("im2")
        im3 = load_png("im3")
        return im1, im2, im3

    @staticmethod
    def _random_crop_3(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, size: int):
        # x*: [3,H,W]
        _, H, W = x1.shape
        if size is None or size == 0 or H < size or W < size:
            return x1, x2, x3
        top  = random.randint(0, H - size)
        left = random.randint(0, W - size)
        s, eH, eW = (slice(top, top+size), slice(left, left+size))
        return x1[:, s, eW], x2[:, s, eW], x3[:, s, eW]

    @staticmethod
    def _center_crop_3(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, size: int):
        if size is None or size == 0:
            return x1, x2, x3
        _, H, W = x1.shape
        if H < size or W < size:
            return x1, x2, x3
        top  = (H - size) // 2
        left = (W - size) // 2
        s, eH, eW = (slice(top, top+size), slice(left, left+size))
        return x1[:, s, eW], x2[:, s, eW], x3[:, s, eW]

    @staticmethod
    def _hflip_3(x1, x2, x3):
        return torch.flip(x1, dims=[2]), torch.flip(x2, dims=[2]), torch.flip(x3, dims=[2])

    def _pack(self, A: torch.Tensor, B: torch.Tensor, tgt: torch.Tensor, tgt_time: float):
        frames = torch.stack([A, B], dim=0)  # [2,3,H,W]
        anchor_times = torch.tensor([0.0, 1.0], dtype=torch.float32)
        target_time  = torch.tensor(tgt_time, dtype=torch.float32)
        return frames, anchor_times, target_time, tgt

    def __getitem__(self, idx: int):
        rel = self.items[idx]  # e.g. "00001/0389"
        im1, im2, im3 = self._load_triplet(rel)

        # Train/eval cropping
        if self.is_train and self.crop_size:
            im1, im2, im3 = self._random_crop_3(im1, im2, im3, self.crop_size)
            if self.aug_flip and random.random() < 0.5:
                im1, im2, im3 = self._hflip_3(im1, im2, im3)
        elif (not self.is_train) and self.crop_size and self.center_crop_eval:
            im1, im2, im3 = self._center_crop_3(im1, im2, im3, self.crop_size)

        if self.mode == "interp":
            # (im1, im3) -> im2
            return self._pack(im1, im3, im2, 0.5)
        elif self.mode == "extrap_fwd":
            # (im1, im2) -> im3
            return self._pack(im1, im2, im3, 2.0)
        elif self.mode == "extrap_bwd":
            # (im2, im3) -> im1
            return self._pack(im2, im3, im1, -1.0)
        else:  # mix
            r = random.random()
            if r < 0.5:
                return self._pack(im1, im3, im2, 0.5)
            elif r < 0.75:
                return self._pack(im1, im2, im3, 2.0)
            else:
                return self._pack(im2, im3, im1, -1.0)


def vimeo_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Batches tuples from Vimeo90KTriplet into:
      frames:       [B,2,3,H,W]
      anchor_times: [B,2]
      target_time:  [B]
      target:       [B,3,H,W]
    """
    frames       = torch.stack([b[0] for b in batch], 0)
    anchor_times = torch.stack([b[1] for b in batch], 0)
    target_time  = torch.stack([b[2] for b in batch], 0)
    target       = torch.stack([b[3] for b in batch], 0)
    return frames, anchor_times, target_time, target
