from pathlib import Path
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class TourismDataset(Dataset):
    """
    returns a normalized image and it's associated pose, intrnsics and scene bound
    """
    def __init__(self, imagepaths, poses, kinvs, bounds):
        """
        Args:
            imagepaths: list of image paths for a scene
            poses (N, 4, 4): camera extrinsic matrices
            kinvs (N, 3, 3): inverse camera intrinsic matrices
            bounds (N, 2): near and far scene bounds
        """
        super().__init__()
        self.imagepaths = imagepaths
        self.poses = poses
        self.kinvs = kinvs
        self.bounds = bounds
        
    def __getitem__(self, idx):
        image = imageio.imread(self.imagepaths[idx])
        image = image[..., :3]/255.
        image = torch.as_tensor(image, dtype=torch.float)

        scale = 0.05
        pose = self.poses[idx]
        pose = torch.as_tensor(pose, dtype=torch.float)
        pose = torch.cat([pose[:3, :3], pose[:3, 3:4]*scale], dim=-1)

        kinv = self.kinvs[idx]
        kinv = torch.as_tensor(kinv, dtype=torch.float)
        
        bound = self.bounds[idx]
        bound = torch.as_tensor(bound, dtype=torch.float)
        bound = bound * torch.as_tensor([0.9, 1.2]) * scale

        return image, pose, kinv, bound
    
    def __len__(self):
        return len(self.imagepaths)


def build_tourism(image_set, args):
    """
    image_set: whether to return train, val or test dataset
    """
    all_imagepaths = sorted(Path(args.image_folder).glob("*.jpg"))
    all_poses = np.load(args.c2w_path)
    all_kinvs = np.load(args.kinv_path)
    all_bounds = np.load(args.bound_path)

    # first 20 images are test, next 5 for validation and the rest for training.
    # https://github.com/tancik/learnit/issues/3
    splits = {
        "test": (all_imagepaths[:20], all_poses[:20], all_kinvs[:20], all_bounds[:20]),
        "val": (all_imagepaths[20:25], all_poses[20:25], all_kinvs[20:25], all_bounds[20:25]),
        "train": (all_imagepaths[25:], all_poses[25:], all_kinvs[25:], all_bounds[25:])
    }

    imagepaths, poses, kinvs, bounds = splits[image_set]
    dataset = TourismDataset(imagepaths, poses, kinvs, bounds)

    return dataset