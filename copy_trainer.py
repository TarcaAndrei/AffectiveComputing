import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, DistilBertModel
import numpy as np
from VisionTransformer import VisionTransformer, vit_small_patch16_224
from torch.utils.data import DataLoader
from VisionTransformer import CrossAttention
from dataset_emotic import DatasetEmotic, CustomCollateFn
from torch.nn import Linear
from torch.nn.functional import softmax



class TransformerTrainer(pl.LightningModule):
    def __init__(self,  text_encoder='distilbert-base-uncased'):
        super(TransformerTrainer, self).__init__()
        # Initialize transformer models
        self.tokenizer_text = AutoTokenizer.from_pretrained(text_encoder)
        self.encoder_text = DistilBertModel.from_pretrained(text_encoder).to(self.device)  


        # Initialize vision transformers encoders 
        self.encoder_vit_image_1 = vit_small_patch16_224(False)
        self.encoder_vit_image_2 = vit_small_patch16_224(False)
        pretrained_weights  = torch.load("dino_deitsmall16_pretrain.pth")
        self.encoder_vit_image_1.load_state_dict(pretrained_weights, strict=False)
        self.encoder_vit_image_2.load_state_dict(pretrained_weights, strict=False)
        # for param in self.encoder_text.parameters():
        #     param.requires_grad = False
        # for param in self.encoder_vit_image_1.parameters():
        #     param.requires_grad = False
        # for param in self.encoder_vit_image_2.parameters():
        #     param.requires_grad = False

        #Cross attention
        self.cross_attention = CrossAttention(dim=384)
        self.cross_attention_2 = CrossAttention(dim=384)

        self.classifier = Linear(384, 26)


    def forward(self, text, img_1, img_2):
        all_input = self.tokenizer_text(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = all_input["input_ids"].to(self.device)
        attention_mask = all_input["attention_mask"].to(self.device)
        text_embeds = self.encoder_text(input_ids, attention_mask)[0]

        output_img_1 = self.encoder_vit_image_1.forward_features(img_1)[:,1:,:]
        output_img_2 = self.encoder_vit_image_2.forward_features(img_2)[:,1:,:]

        visual_cross_attention = self.cross_attention.forward_two(output_img_1, output_img_2)
        visual_cross_attention = visual_cross_attention.repeat((1, text_embeds.shape[1], 1))
        full_output = self.cross_attention_2.forward_two(text_embeds, visual_cross_attention)

        full_output = full_output.flatten(start_dim=1)
        full_output = self.classifier(full_output)
        probs = softmax(full_output, dim=1)
        return probs

    def training_step(self, batch, batch_idx):
        images, crops, labels, texts = batch
        outputs = self.forward(texts, images, crops)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    dataset = DatasetEmotic(dataset_type="train", subset_dim=64)
    dataloader = DataLoader(dataset=dataset, batch_size=8, collate_fn=CustomCollateFn(device=device), drop_last=True, num_workers=4, pin_memory=True)
    model = TransformerTrainer().to(device)
    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(model, dataloader)
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
