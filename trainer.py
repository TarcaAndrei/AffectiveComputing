import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, DistilBertModel, BertModel
import numpy as np
from VisionTransformer import VisionTransformer, vit_small_patch16_224, vit_base_patch16_224
from torch.utils.data import DataLoader, Subset
from VisionTransformer import CrossAttention
from dataset_emotic import CustomCollateFn
from torch.nn import Linear, BatchNorm1d
from torch.nn.functional import softmax
from dataset_emotic import DatasetEmotic
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from collections import Counter

class_frequencies = torch.tensor([346, 108, 107, 2255, 139, 1453, 187, 1063, 259, 381, 133, 2621, 664, 2253, 349, 144, 2219, 85, 1191, 1752, 189, 269, 246, 359, 664, 537], dtype=torch.float32).to('cuda:0')
# {'Engagement': 2621, 'Anticipation': 2255, 'Excitement': 2253, 'Happiness': 2219, 'Pleasure': 1752, 'Confidence': 1453, 'Peace': 1191, 'Disconnection': 1063, 'Esteem': 664, 'Sympathy': 664, 'Yearning': 537, 'Doubt/Confusion': 381, 'Surprise': 359, 'Fatigue': 349, 'Affection': 346, 'Sensitivity': 269, 'Disquietment': 259, 'Suffering': 246, 'Sadness': 189, 'Disapproval': 187, 'Fear': 144, 'Aversion': 139, 'Embarrassment': 133, 'Anger': 108, 'Annoyance': 107, 'Pain': 85}
# class_weights = 1 / class_frequencies * 1000
total_samples = class_frequencies.sum().item()
num_classes = len(class_frequencies)
class_weights = torch.tensor([total_samples / (nr_sampl * num_classes) for nr_sampl in class_frequencies], dtype=torch.float32).to("cuda:0")
print(class_weights)
# class_weights *= 10

# class_weights /= class_weights.sum()

NUM_EPOCHS=30
# 46
class TransformerTrainer(pl.LightningModule):
    def __init__(self,  text_encoder='distilbert-base-uncased'):
        super(TransformerTrainer, self).__init__()
        # Initialize transformer models
        self.tokenizer_text = AutoTokenizer.from_pretrained(text_encoder)
        self.encoder_text = DistilBertModel.from_pretrained(text_encoder).to(self.device)  


        # Initialize vision transformers encoders 
        self.encoder_vit_image_1 = vit_base_patch16_224(False)
        self.encoder_vit_image_2 = vit_base_patch16_224(False)
        pretrained_weights  = torch.load("dino_vitbase16_pretrain.pth")
        self.encoder_vit_image_1.load_state_dict(pretrained_weights, strict=False)
        self.encoder_vit_image_2.load_state_dict(pretrained_weights, strict=False)
        for param in self.encoder_text.parameters():
            param.requires_grad = False
        for param in self.encoder_vit_image_1.parameters():
            param.requires_grad = True
        for param in self.encoder_vit_image_2.parameters():
            param.requires_grad = True

        #Cross attention
        self.cross_attention = CrossAttention(dim=768)
        self.cross_attention_2 = CrossAttention(dim=768)
        self.linear1 = Linear(768, 1024)
        self.relu = torch.nn.ReLU()
        self.linear2 = Linear(1024, 512)
        self.classifier = Linear(512, 26)
        self.batch_norm_1 = BatchNorm1d(768)
        # self.downsampling = Linear(768, 384)


    def forward(self, text, img_1, img_2):
        # Tokenize text and get text embeddings
        all_input = self.tokenizer_text(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = all_input["input_ids"].to(self.device)
        attention_mask = all_input["attention_mask"].to(self.device)
        text_embeds = self.encoder_text(input_ids, attention_mask)[0]  # (64, 169, 768)

        # Get visual embeddings
        output_img_1 = self.encoder_vit_image_1.forward_features(img_1)
        output_img_2 = self.encoder_vit_image_2.forward_features(img_2)



        # First cross-attention between image features
        visual_cross_attention = self.cross_attention.forward_two(output_img_1, output_img_2) * 10 # (64, 1, 768)

        # Expand visual_cross_attention to match text_embeds
        # visual_cross_attention = visual_cross_attention.expand(-1, text_embeds.shape[1], -1)  # (64, 169, 768)

        # Second cross-attention between text and image cross-attention
        full_output = self.cross_attention_2.forward_two(text_embeds, visual_cross_attention)
        # output_img_2 = output_img_2[:, 0, :]
        # Classification
        # full_output = visual_cross_attention.flatten(start_dim=1)
        full_output, _ = torch.max(full_output, dim=1) 
        full_output = self.classifier(self.relu(self.linear2(self.relu(self.linear1(full_output)))))
        return full_output


    def validation_step(self, batch, batch_idx):
        images, crops, labels, texts = batch
        outputs = self.forward(texts, images, crops)

        # Apply sigmoid to outputs

        # Eliminate class 11 with a probability of 0.8 when another class is present
        # mask_class_11 = labels[:, 11] == 1
        # other_classes_mask = labels.sum(dim=1) > 1

        # # Randomly drop class 11 with a probability of 0.8
        # drop_mask = torch.rand(labels.size(0)) < 0.8
        # mask_to_drop = mask_class_11.to(self.device) & other_classes_mask.to(self.device) & drop_mask.to(self.device)
        # new_labels = labels
        # new_labels[mask_to_drop, 11] = 0
        # # new_labels = new_labels * 0.9 + 0.05
        # first_one_indices = (new_labels.cumsum(dim=1) == 1).float()

        # # Apply the mask to keep only the first occurrence of 1
        # new_labels = new_labels * first_one_indices
        new_labels = labels * 0.9 + 0.05
        # Calculate binary cross-entropy loss with class weights
        weights = class_weights.to(labels.device)  # Move weights to the correct device
        loss = F.cross_entropy(outputs, new_labels.float(), weight=weights)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        outputs = torch.nn.functional.softmax(outputs, dim=1)

        predictions = torch.argmax(outputs, dim=1)
        labels = torch.argmax(new_labels, dim=1)

        correct = (predictions == labels).float()
        acc = correct.mean()

        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        images, crops, labels, texts = batch
        outputs = self.forward(texts, images, crops)

        # Apply sigmoid to outputs

        # Eliminate class 11 with a probability of 0.8 when another class is present
        # mask_class_11 = labels[:, 11] == 1
        # other_classes_mask = labels.sum(dim=1) > 1

        # # Randomly drop class 11 with a probability of 0.8
        # drop_mask = torch.rand(labels.size(0)) < 0.8
        # mask_to_drop = mask_class_11.to(self.device) & other_classes_mask.to(self.device) & drop_mask.to(self.device)
        # new_labels = labels
        # new_labels[mask_to_drop, 11] = 0
        # first_one_indices = (labels.cumsum(dim=1) == 1).float()

        # # Apply the mask to keep only the first occurrence of 1
        # new_labels = labels * first_one_indices
        new_labels = labels * 0.9 + 0.05
        # Calculate binary cross-entropy loss with class weights
        weights = class_weights.to(labels.device)  # Move weights to the correct device
        # labels = torch.argmax(new_labels, dim=1)
        loss = F.cross_entropy(outputs, new_labels, weight=weights)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        outputs = torch.nn.functional.softmax(outputs, dim=1)

        # Accuracy calculation
        predictions = torch.argmax(outputs, dim=1)
        labels = torch.argmax(new_labels, dim=1)

        correct = (predictions == labels).float()
        acc = correct.mean()

        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        all_parameters = (
            list(self.encoder_vit_image_1.parameters()) +
            list(self.encoder_vit_image_2.parameters()) +
            list(self.cross_attention.parameters()) +
            list(self.cross_attention_2.parameters()) + 
            list(self.linear1.parameters()) + 
            list(self.linear2.parameters()) +
            list(self.classifier.parameters()) +
            list(self.encoder_text.parameters())
        )
        optimizer = torch.optim.AdamW(all_parameters, lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=NUM_EPOCHS * 956, power=1.0)
        return optimizer


if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Initialize the full dataset
    full_dataset = DatasetEmotic(dataset_type="train")  # Example subset dimension

    # Create subsets
    train_dataset = full_dataset.get_subset("train")
    val_dataset = full_dataset.get_subset("val")
    test_dataset = full_dataset.get_subset("test")

    # dataset = DatasetEmotic(dataset_type="train", subset_dim=32)
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, collate_fn=CustomCollateFn(device=device), drop_last=True)
    dataloader_val = DataLoader(dataset=val_dataset, batch_size=16, collate_fn=CustomCollateFn(device=device), drop_last=True)

    checkpoint_callback = ModelCheckpoint( monitor='val_acc', filename='simple-model-{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}-{train_acc:.2f}-{val_acc:.2f}', save_top_k=1, mode='max')
    checkpoint_path = "/teamspace/studios/this_studio/lightning_logs/version_164/checkpoints/simple-model-epoch=04-train_loss=13.35-val_loss=13.84-train_acc=0.08-val_acc=0.01.ckpt"
    # checkpoint_path = "/teamspace/studios/this_studio/lightning_logs/version_165/checkpoints/simple-model-epoch=04-train_loss=13.73-val_loss=13.71-train_acc=0.01-val_acc=0.01.ckpt"
    model = TransformerTrainer.load_from_checkpoint(checkpoint_path).to(device)
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, callbacks=[checkpoint_callback])


    def get_class_distribution(dataset, num_classes=26):
        """
        Function to get the class distribution of the dataset.
        Args:
        - dataset: The dataset to calculate class distribution.
        - num_classes: The total number of classes (default is 26).
        Returns:
        - A list containing counts for each class (0 to num_classes).
        """
        class_counts = np.zeros(num_classes, dtype=int)

        for i, (_, _, labels, _) in enumerate(dataset):
            print(i)
            labels = labels.cpu().numpy()  # Convert to NumPy array
            class_counts += np.sum(labels, axis=0).astype(np.int32)  # Vectorized sum along axis 0
            print(class_counts)

        return class_counts.tolist()


    # # Example usage:
    # train_class_distribution = get_class_distribution(dataloader_train)
    # val_class_distribution = get_class_distribution(dataloader_val)
    # print(train_class_distribution)
    # print(val_class_distribution)

    trainer.fit(model, val_dataloaders=dataloader_val, train_dataloaders=dataloader_train)
    # trainer.fit(model, train_dataloaders=dataloader_train)
    # trainer.save_checkpoint("first_model.ckpt")

# # Example usage:
# if __name__ == '__main__':
#     from torch.utils.data import DataLoader, TensorDataset

#     # Dummy data for demonstration purposes
#     input_ids_bert = torch.randint(0, 100, (32, 64))
#     attention_mask_bert = torch.ones_like(input_ids_bert)
#     input_ids_roberta = torch.randint(0, 100, (32, 64))
#     attention_mask_roberta = torch.ones_like(input_ids_roberta)
#     labels = torch.randint(0, 2, (32,))

#     dataset = TensorDataset(input_ids_bert, attention_mask_bert, input_ids_roberta, attention_mask_roberta, labels)
#     dataloader = DataLoader(dataset, batch_size=8)

#     model = TransformerTrainer()
#     trainer = pl.Trainer(max_epochs=1)
#     trainer.fit(model, dataloader)

######################################################################### SPLITTED DATASET #############################################

#if __name__ == "__main__":
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the full dataset
#    full_dataset = DatasetEmotic(dataset_type="train", subset_dim=1000)  # Example subset dimension

    # Create subsets
#    train_dataset = full_dataset.get_subset("train")
#    val_dataset = full_dataset.get_subset("val")
#    test_dataset = full_dataset.get_subset("test")

    # Dataloaders
#    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=CustomCollateFn(device=device), drop_last=True)
#    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=CustomCollateFn(device=device), drop_last=False)
#    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=CustomCollateFn(device=device), drop_last=False)

    # Initialize model
#    model = TransformerTrainer().to(device)

    # Trainer setup
#    trainer = pl.Trainer(max_epochs=100)

    # Train and validate
#    trainer.fit(model, train_loader, val_loader)

    # Test the model
#    trainer.test(model, test_loader)