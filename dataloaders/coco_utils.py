import copy
import os

import random
import torch
import torch.utils.data
import torchvision
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import torchvision.transforms as T
from torchvision.transforms import functional as F

random.seed(1)


class FilterAndRemapCocoCategories:
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset
    
class RandomResize:
    def __init__(self, min_size, max_size=None, same_size=False):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
        self.same = same_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        if self.same: size = (size, size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target

class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target
        
class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
        
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

def get_coco(root, image_set, same_size):

    trans = Compose([
                RandomResize(400, 400, same_size),
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                
    PATHS = {
        #"train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": ('val2017', 'instances_val2017.json'),
    }
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    transforms = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask(), trans])

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset

def get_coco_corrupted(corrupted_root, root, image_set, same_size):

    trans = Compose([
                RandomResize(400, 400, same_size),
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                
    PATHS = {
        #"train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": (corrupted_root, 'instances_val2017.json'),
    }
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    transforms = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask(), trans])

    img_folder, ann_file = PATHS[image_set]
    # img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset