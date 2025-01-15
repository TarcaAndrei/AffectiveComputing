import numpy as np
from VisionTransformer import VisionTransformer
from VisionTransformer import CrossAttention
import torch

image = torch.tensor(np.expand_dims(np.transpose(np.load('/teamspace/studios/this_studio/img_arrs/crop_arr_test_0.npy'), axes=[2, 0, 1]), axis=0)).type(torch.float32)

vision_transformer1 = VisionTransformer()
outputs1 = vision_transformer1.forward_features(image)
print(outputs1.shape)
vision_transformer2 = VisionTransformer()
outputs2 = vision_transformer2.forward_features(image)
print(outputs2.shape)

cross_attention = CrossAttention(dim=768)
outputs = cross_attention.forward_two(outputs1, outputs2)
print(outputs.shape)