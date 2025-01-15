from dataset_emotic import  DatasetEmotic, emotic_classses
import torch
import json
full_dataset = DatasetEmotic(dataset_type="train", split_ratios=(1, 0, 0))  # Example subset dimension


all_classes = {}

distributii = {'Anticipation': 5236, 'Engagement': 7500, 'Confidence': 2874, 'Happiness': 2122, 'Excitement': 1731, 'Disconnection': 1310, 'Affection': 1068, 'Peace': 702, 'Yearning': 445, 'Pleasure': 375, 'Annoyance': 287, 'Esteem': 272, 'Doubt/Confusion': 271, 'Fatigue': 224, 'Disquietment': 219, 'Anger': 201, 'Sensitivity': 155, 'Sadness': 150, 'Disapproval': 129, 'Sympathy': 115, 'Surprise': 113, 'Aversion': 94, 'Pain': 73, 'Fear': 73, 'Embarrassment': 71, 'Suffering': 18}
distributii = dict(sorted(distributii.items(), key=lambda item: item[1], reverse=True))
# print(distributii)

matrix = {}


new_distribution = {}
for i in range(len(full_dataset)):
    _, bboxes, _ = full_dataset[i]
    lista_clase_obiecte = {}
    for obje in bboxes:
        all_classes_image = obje['labels']
        for element in distributii:
            if len(all_classes_image) == 1:
                break
            if element in all_classes_image:
                all_classes_image.remove(element)

        label = all_classes_image[0]
        if label not in all_classes:
            all_classes[label] = 0 
        all_classes[label] += 1
        lista_clase_obiecte[len(lista_clase_obiecte)] = label
    matrix[i] = lista_clase_obiecte


for i, el in enumerate(matrix):
    if i == 100:
        break
    # print(matrix[i])

with open("fisier.json", "w") as file_reader:
    json.dump(matrix, file_reader)

print(dict(sorted(all_classes.items(), key=lambda item: item[1], reverse=True)))





    # for obje in bboxes:
    #     all_classes_image = obje['labels']
    #     class_indexes = [emotic_classses.index(elem) for elem in all_classes_image]
    #     more_hot_encodings = [1.0 if i in class_indexes else 0 for i in range(len(emotic_classses))]
    #     labels = torch.tensor(more_hot_encodings).unsqueeze(0)
    #     mask_class_11 = labels[:, 11] == 1
    #     other_classes_mask = labels.sum(dim=1) > 1

    #     # Randomly drop class 11 with a probability of 0.8
    #     drop_mask = torch.rand(labels.size(0)) < 1
    #     mask_to_drop = mask_class_11 & other_classes_mask & drop_mask
    #     new_labels = labels
    #     new_labels[mask_to_drop, 11] = 0
    #     class_id  = torch.where(new_labels == 1)[-1][0].item()
    #     label = emotic_classses[class_id]
    #     if label not in all_classes:
    #         all_classes[label] = 0 
    #     all_classes[label] += 1