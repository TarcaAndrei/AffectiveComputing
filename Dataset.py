from pytorch_lightning import LightningDataModule
import pandas as pd
import numpy as np
from pathlib import Path

class DatasetEmotic(LightningDataModule):
    def __init__(self, csv_path, img_dir):
        self.data_frame = pd.read_csv(csv_path)
        self.csv_path = csv_path
        self.img_dir = Path(img_dir)
        self.classes = [
            "Valence", "Arousal", "Dominance", "Peace", "Affection", "Esteem", 
            "Anticipation", "Engagement", "Confidence", "Happiness", "Pleasure", 
            "Excitement", "Surprise", "Sympathy", "Doubt/Confusion", "Disconnection", 
            "Fatigue", "Embarrassment", "Yearning", "Disapproval", "Aversion", 
            "Annoyance", "Anger", "Sensitivity", "Sadness", "Disquietment", "Fear", 
            "Pain", "Suffering"
        ]

    def _get_image(self, path):
        return np.load(self.img_dir / path)

    def _get_clase(self, data):
        clase_active = []
        for clasa in self.classes:
            if data[clasa] == 1.0:
                clase_active.append(clasa) 
        return clase_active

    def __getitem__(self, idx: int):
        data = self.data_frame.iloc[idx]
        full_img = self._get_image(data["Arr_name"])
        crop_img = self._get_image(data["Crop_name"])
        clasele = self._get_clase(data)
        # print(full_img.shape)
        # print(crop_img.shape)
        print(clasele)
        