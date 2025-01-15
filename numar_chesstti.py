# import json


# with open("fisier_nou.json", "r") as file_reader:
#     matrix = json.load(file_reader)

# noile_clase = {}

# for _, val in matrix.items():
#     for _, clasa in val.items():
#         if clasa not in noile_clase:
#             noile_clase[clasa] = 0
#         noile_clase[clasa] += 1

# noile_clase = dict(sorted(noile_clase.items(), key=lambda item: item[1], reverse=True))
# print(noile_clase)

from dataset_emotic import  DatasetEmotic, emotic_classses

full_dataset = DatasetEmotic(dataset_type="train")  # Example subset dimension

for el in full_dataset:
    print(el)