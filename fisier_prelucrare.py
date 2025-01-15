from dataset_emotic import  DatasetEmotic, emotic_classses
import torch
import random
import json
full_dataset = DatasetEmotic(dataset_type="train", split_ratios=(1, 0, 0))  # Example subset dimension

with open("fisier.json", "r") as file_reader:
    matrix = json.load(file_reader)


new_matrix = {}

for idx, val in matrix.items():
    for idxx, valll in val.items():
        if valll == "Engagement":
            if(random.random()) > 0.5:
                if idx not in new_matrix:
                    new_matrix[idx] = {
                    }
                new_matrix[idx][idxx] = valll

        elif valll == "Excitement":
            if(random.random()) > 0.3:
                if idx not in new_matrix:
                    new_matrix[idx] = {
                    }
                new_matrix[idx][idxx] = valll
        else:
            if idx not in new_matrix:
                new_matrix[idx] = {}
            new_matrix[idx][idxx] = valll


with open("fisier_nou.json", "w") as file_reader:
    json.dump(new_matrix, file_reader)


