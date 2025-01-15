from __future__ import annotations
from pytorch_lightning import LightningDataModule
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io

class DatasetEmotic(LightningDataModule):
    def __init__(self, mat_file_path: str = "annots_arrs/Full_Annotations_Fixed_Val.mat", root_dir_imgs: str = "img_arrs", dataset_type: str="val"):
    # def __init__(self, mat_file_path: str = "annots_arrs/Modified_Annotations.mat", root_dir_imgs: str = "img_arrs", dataset_type: str="train"):
        self.dataset_type = dataset_type
        self.data_frame = pd.read_csv("annots_arrs/annot_arrs_val.csv")
        self.data_frame_2 = pd.read_csv("annots_arrs/annot_arrs_train.csv")
        self.root_dir_imgs = Path(root_dir_imgs)
        self.start = 0
        self.mat_data = scipy.io.loadmat(mat_file_path)
        # mat_file = h5py.File(mat_file_path, "r")
        # with h5py.File(mat_file_path, "r+") as mat_file:
        # self.mat_data = math_file
    
    def _get_image(self, annotation):
        npy_file = self._search_npy_from_img(annotation[0][0])
        return npy_file
        # print(npy_file)
        # return np.load(self.root_dir_imgs / npy_file)
    
    def _get_persons(self, initial_img, persons) -> list[dict]:
        all_persons = []
        for person in persons:
            bbox = person[0][0]
            labels_list = person[1][0][0][0][0]
            cropped_img = self._crop_img(initial_img, bbox)
            all_persons.append({
                "bbox": bbox,
                "labels": labels_list,
                "cropped_img": cropped_img
            })
        return all_persons

    def _search_npy_from_img(self, img_file: str) -> str:
        for i in range(0, len(self.data_frame)):
            if self.data_frame.iloc[i]["Filename"] == img_file:
                return self.data_frame.iloc[i]["Arr_name"]
        for i in range(0, len(self.data_frame_2)):
            if self.data_frame_2.iloc[i]["Filename"] == img_file:
                return self.data_frame_2.iloc[i]["Arr_name"]
        raise ValueError(f"Nu s-a gasit! {img_file}")

    def _crop_img(self, initial_img, crop_bbox):
        x_min, x_max, y_min, y_max = crop_bbox
        return initial_img[y_min:y_max, x_min:x_max]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list[dict]]:
        if idx > self.__len__():
            raise IndexError("Dataset out of bounds!")
        annotation = self.mat_data[self.dataset_type][0][idx]
        img_path = self._get_image(annotation)
        self.mat_data[self.dataset_type][0][idx][1] = np.array([img_path], dtype='<U50')
        # persons = annotation[4][0]
        # list_bboxes = self._get_persons(image, persons)
        # return image, list_bboxes

    def __len__(self) -> int:
        return len(self.mat_data[self.dataset_type][0])
    
    def save_annotations(self):
        output_path = 'annots_arrs/Full_Annotations_Fixed_Val_Test.mat'  # Path to save the modified file
        scipy.io.savemat(output_path, self.mat_data)


        

