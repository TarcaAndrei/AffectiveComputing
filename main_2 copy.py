import json
import torch
import time
from dataset_emotic import DatasetEmotic, CustomCollateFn
from PIL import Image


dataset = DatasetEmotic(dataset_type="test")
dataset_test = DatasetEmotic(dataset_type="test")

for el in dataset:
    print(el)
    break

for el in dataset_test:
    print(el)
    break