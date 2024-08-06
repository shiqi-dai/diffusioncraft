import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from worldgan.minecraft.special_minecraft_downsampling import special_minecraft_downsampling

def load_data(
    *,
    opt,
    mc_level,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    stop_scale=16,
    current_scale=0,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    dataset = MCDataset(
        opt,
        mc_level,
        image_size,
        data_dir,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        stop_scale=stop_scale,
        current_scale=current_scale,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class MCDataset(Dataset):
    def __init__(
        self,
        opt,
        level,
        resolution,
        image_path,
        classes=None,
        shard=0,
        num_shards=1,
        current_scale=0,
        stop_scale=16,
    ):
        super().__init__()
        self.opt = opt
        use_hierarchy = False if self.opt.repr_type else True
        scaled_list = special_minecraft_downsampling(self.opt.num_scales, self.opt.scale_info, level, self.opt.token_list, use_hierarchy)
        reals = [*scaled_list, level] 
        self.reals = reals[current_scale - 1 : stop_scale] 
        self.reals = [real.squeeze(dim=0) for real in self.reals]
        print("Scaled Shapes:")
        for r in self.reals:
            print(r.shape) 

    def __len__(self):
        return 10000
        # return len(self.reals)  cannot start training

    def __getitem__(self, idx): 
        arr = self.reals[-1] 
        out_dict = {}
        return arr, out_dict
