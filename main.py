# from Dataset import DatasetEmotic

# dataset = DatasetEmotic("annots_arrs/annot_arrs_train.csv", "img_arrs")

# # print(dataset[0])

# for i in range(10):
#     dataset[i]

import scipy.io

# Load the .mat file
mat_data = scipy.io.loadmat('annots_arrs/Annotations.mat')

# The 'Annotations' field should be a list containing train, val, test sets
train_set = mat_data["train"][0]

# To access the annotations for a specific set (e.g., train)

# Accessing the first image's annotations in the 'train' set
first_image_annotation = train_set[0]
img_file = first_image_annotation[0][0]
persons = first_image_annotation[4][0]

for person in persons:
    bbox = person[0][0]
    labels = person[1][0][0][0][0]
    pass

# Access metadata for the first image
filename = first_image_annotation['filename'][0]
folder = first_image_annotation['folder'][0]

# Accessing the image size (row x column format)
image_size = first_image_annotation['image_size'][0]

# Accessing the 'person' data for the first image (assuming there are two persons annotated)
person_1 = first_image_annotation['person'][0][0]  # Person 1 annotations

# Access bounding box for the first person
body_bbox = person_1['body_bbox'][0]  # [x1, y1, x2, y2]

# Access emotional categories for the first person
annotations_categories = person_1['annotations_categories'][0]
emotion_categories = annotations_categories['categories'][0]

# Access continuous annotations for the first person (valence, arousal, dominance)
annotations_continuous = person_1['annotations_continuous'][0]
valence = annotations_continuous['valence'][0]
arousal = annotations_continuous['arousal'][0]
dominance = annotations_continuous['dominance'][0]

# Accessing demographic information
gender = person_1['gender'][0]
age = person_1['age'][0]

# Printing some information
print(f"Filename: {filename}")
print(f"Folder: {folder}")
print(f"Image Size: {image_size}")
print(f"Bounding Box (Person 1): {body_bbox}")
print(f"Emotion Categories: {emotion_categories}")
print(f"Continuous Annotations - Valence: {valence}, Arousal: {arousal}, Dominance: {dominance}")
print(f"Gender: {gender}")
print(f"Age: {age}")
