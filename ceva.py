from dataset_emotic import emotic_classses


ceva = {'Engagement': 2621, 'Anticipation': 2255, 'Excitement': 2253, 'Happiness': 2219, 'Pleasure': 1752, 'Confidence': 1453, 'Peace': 1191, 'Disconnection': 1063, 'Esteem': 664, 'Sympathy': 664, 'Yearning': 537, 'Doubt/Confusion': 381, 'Surprise': 359, 'Fatigue': 349, 'Affection': 346, 'Sensitivity': 269, 'Disquietment': 259, 'Suffering': 246, 'Sadness': 189, 'Disapproval': 187, 'Fear': 144, 'Aversion': 139, 'Embarrassment': 133, 'Anger': 108, 'Annoyance': 107, 'Pain': 85}


new_dict = {}

for el, val in ceva.items():
    new_dict[emotic_classses.index(el)] = val

new_dict = dict(sorted(new_dict.items(), key=lambda item: item[0]))
print(new_dict)

print(new_dict.values())