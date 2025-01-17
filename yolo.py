from ultralytics import YOLO

# Load a model
model = YOLO("weights_yolo.pt")  # load an official model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

print(results)