from __future__ import annotations
from pytorch_lightning import LightningDataModule
from pathlib import Path
import numpy as np
import scipy.io
import json
import random
import torch
from torchvision import transforms
from torch.utils.data import Subset


emotic_classses = [
                "Affection", 
                "Anger",
                "Annoyance",
                "Anticipation",
                "Aversion",
                "Confidence",
                "Disapproval",
                "Disconnection",
                "Disquietment",
                "Doubt/Confusion",
                "Embarrassment",
                "Engagement",
                "Esteem",
                "Excitement",
                "Fatigue",
                "Fear",
                "Happiness",
                "Pain",
                "Peace",
                "Pleasure",
                "Sadness",
                "Sensitivity",
                "Suffering",
                "Surprise",
                "Sympathy",
                "Yearning"
            ]


class DatasetEmotic(LightningDataModule):
    def __init__(self, mat_file_path: str = "annots_arrs/Annotations.mat", root_dir_imgs: str = "img_arrs", json_transcripts: str = "captions.json", dataset_type: str="train", subset_dim: int | None = None, split_ratios=(0.9, 0.1, 0.0)):
        self.dataset_type = dataset_type
        self.root_dir_imgs = Path(root_dir_imgs)
        self.mat_data = scipy.io.loadmat(mat_file_path)[dataset_type][0]
        with open(json_transcripts, "r") as json_file:
            self.text_data = json.load(json_file)[dataset_type]
        if subset_dim is not None:
            self.subset_dim = subset_dim
        else:
            self.subset_dim = len(self.mat_data)
        self.split_ratios = split_ratios
        with open("fisier_nou.json", "r") as file_reader:
            self.matrix = json.load(file_reader)
            self.subset_dim = min(len(self.matrix), self.subset_dim)
        self.mapping = {i: el for i, el in enumerate(self.matrix)}
        self.train_indices, self.val_indices, self.test_indices = self._split_dataset()

    def _get_image(self, annotation) -> np.ndarray:
        img_file = self.root_dir_imgs / annotation[1][0].strip()
        return np.load(img_file)


    def _scale_bounding_boxes(self, bbox, original_width, original_height, target_width=224, target_height=224):
        """
        Scales bounding box coordinates to match a new image resolution.
        
        Parameters:
            bboxes (list of lists): A list of bounding boxes, each defined as [x_min, x_max, y_min, y_max].
            original_width (int or float): Width of the original image.
            original_height (int or float): Height of the original image.
            target_width (int or float): Width of the target image.
            target_height (int or float): Height of the target image.
            
        Returns:
            list of lists: Scaled bounding boxes with coordinates adjusted for the target resolution.
        """
        # Calculate scaling factors
        scale_width = target_width / original_width
        scale_height = target_height / original_height
        x_min, y_min, x_max, y_max = bbox
        scaled_bbox = [
            x_min * scale_width,
            y_min * scale_height,
            x_max * scale_width,
            y_max * scale_height
        ]
        return scaled_bbox
    
    def _get_persons(self, idx: int, initial_img: np.ndarray, persons: list[dict], height: int, width: int) -> list[dict]:
        """
        This function extracts for each person the bbox and the classes
        Returns:
            a list of dict, one for each person, with the following keys:
                'bbox': the bbox scaled in image coordinates in the format X_min, Y_min, X_max, Y_max
                'lables': a list of strings representing the classes
                'cropped_img': np.ndarray representing just the cropped person from the image
        """
        all_persons = []
        for person in persons:
            bbox = person[0][0] #x1 y1 x2 y2
            labels_list = person[1][0][0][0][0]
            bbox = self._scale_bounding_boxes(bbox, width, height)
            cropped_img = self._crop_img(initial_img, bbox)
            if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                raise ValueError("aici")
                print("aici")
            all_persons.append({
                "bbox": bbox,
                "labels": [lbl[0] for lbl in labels_list],
                "cropped_img": cropped_img
            })
        

        # need to filter the extras out
        obiecte_interes = self.matrix[str(idx)]
        remaining_persons = [all_persons[int(index)] for index in obiecte_interes]
        for i, (idxxx, el) in enumerate(obiecte_interes.items()):
            remaining_persons[i]["labels"] = [el]
        return remaining_persons

    def _crop_img(self, initial_img, crop_bbox):
        x_min, y_min, x_max, y_max = [int(c) for c in crop_bbox]
        return initial_img[y_min:y_max, x_min:x_max]

    def _get_img_dims(self, annotation) -> tuple[int, int]:
        """
        Returns the height and the width of the image
        """
        return annotation[2][0][0][1][0][0], annotation[2][0][0][0][0][0]

    def _get_text(self, idx: int) -> str:
        try:
            return self.text_data[str(idx)]
        except KeyError as ke:
            return "test"

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

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list[dict], str]:
        """
        Returns a tuple:
            the image
            a list of dict, one for each person, with the following keys:
                'bbox': the bbox scaled in image coordinates in the format X_min, X_max, Y_min, Y_max
                'lables': a list of strings representing the classes
                'cropped_img': np.ndarray representing just the cropped person from the image
        """
        if idx >= self.__len__():
            raise IndexError("Dataset out of bounds!")
        idx = int(self.mapping[idx]) # real index....
        annotation = self.mat_data[idx]
        image = self._get_image(annotation)
        height, width = self._get_img_dims(annotation)
        caption = self._get_text(idx)
        persons = annotation[4][0]
        try:
            list_bboxes = self._get_persons(idx, image, persons, height, width)
            return image, list_bboxes, caption
        except:
            return self.__getitem__(0)


    def __len__(self) -> int:
        return self.subset_dim

class CustomCollateFn:
    def __init__(self, device, classes :list | None = None):
        self.device = device
        self.classes = classes
        if self.classes is None:
            self.classes = emotic_classses
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

    def __call__(self, batch: list[tuple[np.ndarray, list[dict], str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        this receives a list of batches
        Returns:
            a tuple of 3 Tensors:
                first - the entire images stacked -> [B, C, H, W] (our case - [B, 3, 224, 224])
                second - the cropped images stacked (one person random selected from each initial image and rescaled at 224x224)
                        (same shape as the first one)
                third - the labels, of shape [B, NoClasses] (our case - [B, 26])
                        for each person, we have the soft labels - meaning that:
                                if there is just one class, there will be a 1 on that position
                                if there are 2 or more, the 1 will be divided with the number of classes
                                (for 2 there will be 2 0.50, for 3 -> 0.33 and so on)
                        (there may be more that just one single one)
        """
        all_images = []
        all_crops = []
        all_features = []
        all_texts = []
        for i in range(len(batch)):
            img, lbl, caption = batch[i]
            img = self.transforms(img)
            # here we select only one person randomly
            idx = random.randint(0, len(lbl) - 1)
            person = lbl[idx]
            img_cropped = self.transforms(person["cropped_img"])
            all_crops.append(img_cropped)
            all_images.append(img)
            class_indexes = [self.classes.index(elem) for elem in person["labels"]]
            if len(class_indexes) > 1:
                raise ValueError("nu meerge cum trb")
            more_hot_encodings = [1.0 if i in class_indexes else 0 for i in range(len(self.classes))]
            all_texts.append(caption)
            all_features.append(more_hot_encodings)
        all_images = torch.stack(all_images).to(device=self.device)
        all_crops = torch.stack(all_crops).to(device=self.device)
        all_features = torch.tensor(all_features, device=self.device)
        return all_images, all_crops, all_features, all_texts

