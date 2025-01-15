from trainer import TransformerTrainer
from ultralytics import YOLO
import torch
from torch import tensor
from dataset_emotic import DatasetEmotic, emotic_classses
from torchvision import transforms
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Evaluator:
    def __init__(self, dataset_dim, class_threshold=0.5, yolo_threshold=0.5):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.image_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
            ])
        self.class_threshold = class_threshold

        self.yolo_threshold = yolo_threshold
        # self.model = TransformerTrainer().to(self.device)
        self.detection_model = YOLO("weights_yolo.pt", verbose=False).eval() 
        self.dataset = DatasetEmotic(dataset_type="train", subset_dim=dataset_dim) 
        self.metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
        self.model = TransformerTrainer.load_from_checkpoint("/teamspace/studios/this_studio/lightning_logs/version_75/checkpoints/simple-model-epoch=02-train_loss=0.02-val_loss=0.13-train_acc=0.26-val_acc=0.28.ckpt", text_encoder='distilbert-base-uncased').to(self.device).eval()


    def plot_image(self, image, bboxes, labels, title):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            ax = plt.gca()
            dict_labels = {}

            for bbox, label in zip(bboxes, labels):
                print(label)
                bbox_str = str(bbox)
                if bbox_str in dict_labels:
                    dict_labels[bbox_str] += 10
                else:
                    dict_labels[bbox_str] = 0
                x_min, y_min, x_max, y_max = bbox
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 10  + dict_labels[bbox_str], emotic_classses[label], color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

            plt.title(title)
            plt.axis('off')
            plt.savefig('imagine.png')
            plt.clf()

    @torch.no_grad()
    def compute(self, plot=False):
        for i, (imagine, ground_truth, text) in enumerate(self.dataset):
            imagine_transformed = self.image_transforms(imagine)
            predictie_yolo = self.detection_model(imagine, imgsz=224, verbose=False)
            target_bboxes = []
            target_labels = []
            for person in ground_truth:
                bboxu = person["bbox"]
                clasele = person["labels"]
                for clasa in clasele:
                    target_bboxes.append(bboxu)
                    target_labels.append(emotic_classses.index(clasa))
            target = [
                {
                    "boxes": tensor(target_bboxes),
                    "labels": tensor(target_labels),
                }
            ]
            pred_boxes = []
            pred_scores = []
            pred_labels = []
            for box in predictie_yolo[0].boxes:
                if int(box.cls.item()) == 0 and box.conf.item() > self.yolo_threshold:
                    predicted_box = box.xyxy[0].to("cpu", dtype=int).tolist()
                    x_min, y_min, x_max, y_max = predicted_box
                    cropped_img = imagine[y_min:y_max, x_min:x_max]
                    img_transformed = self.image_transforms(cropped_img)
                    output = self.model.forward(text, imagine_transformed.unsqueeze(0).to(self.device), img_transformed.unsqueeze(0).to(self.device))
                    scores = F.sigmoid(output)
                    predictions = scores > self.class_threshold
                    pred_classes = torch.where(predictions[0])[0].to("cpu").tolist()
                    print(pred_classes)
                    for clasa in pred_classes:
                        pred_boxes.append(predicted_box)
                        pred_labels.append(clasa)
                        pred_scores.append(scores[0][clasa].to("cpu").item())
            preds = [
                {
                    "boxes": tensor(pred_boxes),
                    "scores": tensor(pred_scores),
                    "labels": tensor(pred_labels)
                }
            ]
            if plot:
                print(pred_labels)
                self.plot_image(imagine, pred_boxes, pred_labels, title="Predictions")
                # self.plot_image(imagine, target_bboxes, target_labels, title="Ground Truth")
            self.metric.update(preds, target)
            # if i % 5 == 0:
            print(self.metric.compute())
            print("------------------------------------------------------")
            self.metric.reset()
        return self.metric.compute()

if __name__ == "__main__":
    evalutator = Evaluator(17000)
    result = evalutator.compute(True)
    print(result)

