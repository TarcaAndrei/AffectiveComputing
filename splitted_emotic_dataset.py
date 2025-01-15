from __future__ import annotations
from pytorch_lightning import LightningDataModule
from pathlib import Path
import scipy.io
import json
import random
from torch.utils.data import Subset

class DatasetEmotic(LightningDataModule):
    def __init__(self, mat_file_path="annots_arrs/Annotations.mat", root_dir_imgs="img_arrs", json_transcripts="captions.json", dataset_type="train", subset_dim=None, split_ratios=(0.7, 0.2, 0.1)):
        self.dataset_type = dataset_type
        self.root_dir_imgs = Path(root_dir_imgs)
        self.mat_data = scipy.io.loadmat(mat_file_path)[dataset_type][0]
        with open(json_transcripts, "r") as json_file:
            self.text_data = json.load(json_file)
        # self.subset_dim = len(self.mat_data)
        if subset_dim is not None:
            self.subset_dim = subset_dim
        else:
            self.subset_dim = len(self.mat_data)
        self.split_ratios = split_ratios

        # Indices for dataset splits
        self.train_indices, self.val_indices, self.test_indices = self._split_dataset()

    def _split_dataset(self):
        # Shuffle indices
        indices = list(range(self.subset_dim))
        random.shuffle(indices)

        # Calculate split sizes
        train_size = int(self.split_ratios[0] * len(indices))
        val_size = int(self.split_ratios[1] * len(indices))

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return train_indices, val_indices, test_indices

    def get_subset(self, subset_type):
        if subset_type == "train":
            subset_indices = self.train_indices
        elif subset_type == "val":
            subset_indices = self.val_indices
        elif subset_type == "test":
            subset_indices = self.test_indices
        else:
            raise ValueError("Invalid subset type. Choose from 'train', 'val', or 'test'.")
        return Subset(self, subset_indices)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Dataset out of bounds!")
        annotation = self.mat_data[idx]
        image = self._get_image(annotation)
        height, width = self._get_img_dims(annotation)
        caption = self._get_text(idx)
        persons = annotation[4][0]
        try:
            list_bboxes = self._get_persons(image, persons, height, width)
            return image, list_bboxes, caption
        except:
            return self.__getitem__(idx + 1)

    def __len__(self):
        return self.subset_dim