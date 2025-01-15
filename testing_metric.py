from torch import tensor
from torchmetrics.detection import MeanAveragePrecision
preds = [
   dict(
     boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
     scores=tensor([0.53]),
     labels=tensor([0]),
   )
 ]
target = [
   dict(
     boxes=tensor([[214.0, 41.0, 562.0, 285.0],
                    [214.0, 41.0, 562.0, 285.0]]),
     labels=tensor([1, 0]),
   )
 ]
metric = MeanAveragePrecision(iou_type="bbox")
metric.update(preds, target)
print(metric.compute())