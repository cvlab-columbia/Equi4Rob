from os.path import join, exists

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class VocSegmentationDataset(Dataset):

    def __init__(self, root_dir, phase, transforms):
        self._root_dir = root_dir
        self._phase = self._check_phase(phase)
        self._img_mask_path_pairs = self._get_img_mask_path_pairs()
        self._transforms = transforms
        # self._palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
        #    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
        #    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
        # self._palette = {item: i for i, item in enumerate(self._palette)}
        VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                        [0, 64, 128]]
        self.colormap2label = self.voc_colormap2label(VOC_COLORMAP)

    @staticmethod
    def _check_phase(phase):
        if phase in ['train', 'val']:
            return phase
        else:
            print(phase)
            raise ValueError('Phase must be from ["train", "val"]')

    def _get_img_mask_path_pairs(self):
        if self._phase == 'train':
            return self._get_img_mask_path_pairs_train()
        else:
            return self._get_img_mask_path_pairs_val()

    def _get_img_mask_path_pairs_train(self):
        train_aug_list_path = join(self._root_dir, 'VOC2012/ImageSets/Segmentation/trainaug.txt')
        img_base_path = join(self._root_dir, 'VOC2012/JPEGImages')
        annot_base_path = join(self._root_dir, 'VOC2012/SegmentationClassAug')
        pairs = self._generate_pairs(img_base_path, annot_base_path, train_aug_list_path)
        return pairs

    def _generate_pairs(self, img_base_path, annot_base_path, id_list_path):
        with open(id_list_path) as infile:
            ids = infile.read().strip().split('\n')
        pairs = [self._get_pair(img_base_path, annot_base_path, _id) for _id in ids]
        return pairs

    @staticmethod
    def _get_pair(img_base_path, annot_base_path, _id):
        img_path = join(img_base_path, _id + '.jpg')
        annot_path = join(annot_base_path, _id + '.png')
        return img_path, annot_path

    def _get_img_mask_path_pairs_val(self):
        val_list_path = join(self._root_dir, 'VOC2012/ImageSets/Segmentation/val.txt')
        img_base_path = join(self._root_dir, 'VOC2012/JPEGImages')
        annot_base_path = join(self._root_dir, 'VOC2012/SegmentationClass')
        pairs = self._generate_pairs(img_base_path, annot_base_path, val_list_path)
        return pairs

    @staticmethod
    def voc_colormap2label(voc_colormap):
        """Build the mapping from RGB to class indices for VOC labels."""
        colormap2label = np.zeros(256 ** 3)
        for i, colormap in enumerate(voc_colormap):
            colormap2label[
                (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        return colormap2label

    def __len__(self):
        return len(self._img_mask_path_pairs)

    def __getitem__(self, index):
        img_path, mask_path = self._img_mask_path_pairs[index]
        img = np.asarray(Image.open(img_path), np.uint8)
        img = cv2.resize(img, (480, 480))
        mask = self._prepare_mask_train(mask_path) if self._phase == 'train' else self._prepare_mask_val(mask_path)
        mask = cv2.resize(mask, (480, 480), interpolation=cv2.INTER_NEAREST)
        data = [img, mask]
        data = [torch.Tensor(item) for item in data]
        data = [data[0].permute(2, 0, 1).type(torch.float32), data[1].type(torch.long)]
        # data = [Image.fromarray(item) for item in data]
        # if self._transforms is not None:
        #     data = list(self._transforms(*data))
        return tuple(data + [img_path])

    def _prepare_mask_val(self, mask_path):
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        return self.voc_label_indices(mask)
        # new_mask = np.zeros(mask.shape[:-1], np.uint8)
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         # if mask[i, j].all() != 0:
        #         #     print(mask[i, j])
        #         new_mask = self._palette.get(tuple(mask[i, j]), 0)
        # return new_mask

    def voc_label_indices(self, mask):
        """Map any RGB values in VOC labels to their class indices."""
        mask = mask.astype(np.int32)
        idx = ((mask[:, :, 0] * 256 + mask[:, :, 1]) * 256
               + mask[:, :, 2])
        return self.colormap2label[idx]

    @staticmethod
    def _prepare_mask_train(mask_path):
        if not exists(mask_path):
            mask_path = mask_path.replace('SegmentationClassAug', 'SegmentationClass')
        return np.array(Image.open(mask_path), np.uint8)


if __name__ == "__main__":
    dataset = VocSegmentationDataset('/home/jbnerd/data/VOCdevkit', 'val', None)
    for i, (img, mask, file) in enumerate(dataset):
        mask = np.array(mask)
        print(file)
        print(mask.shape)
        print(np.unique(mask))
        print(np.min(mask), np.max(mask))
        print(mask)
        # cv2.imshow('Test', np.array(img))
        # cv2.waitKey(0)
        # cv2.imshow('Test', np.array(mask))
        # cv2.waitKey(0)
        if i == 5:
            break