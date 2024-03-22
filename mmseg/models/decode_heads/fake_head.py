from .decode_head import BaseDecodeHead
import torch
from torch.nn.functional import sigmoid
from mmseg.registry import MODELS

@MODELS.register_module()
class FakeHead(BaseDecodeHead):
    """Fake DecodeHead for BackboneOlySegmenter.

    Args:
        BaseDecodeHead (): 
    """
    def __init__(self, **kwargs):
        super().__init__(input_transform=None, **kwargs)
        self.fake = None

    def cross_entropy_loss_RCF(prediction, label):
        label = label.long()
        # label2 = label.float()
        mask = label.float()
        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0
        prediction = sigmoid(prediction)
        # print(label.shape)
        cost = torch.nn.functional.binary_cross_entropy(
                prediction.float(),label.float(),weight = mask, reduce=False)
        # weight = mask
        # return torch.sum(cost)
        # loss = F.binary_cross_entropy(prediction, label2, mask, size_average=True)
        return torch.sum(cost) /(num_positive+num_negative)
    
    # def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
    #          train_cfg: ConfigType) -> dict:
    #     """Forward function for training.

    #     Args:
    #         inputs (Tuple[Tensor]): List of multi-level img features.
    #         batch_data_samples (list[:obj:`SegDataSample`]): The seg
    #             data samples. It usually includes information such
    #             as `img_metas` or `gt_semantic_seg`.
    #         train_cfg (dict): The training config.

    #     Returns:
    #         dict[str, Tensor]: a dictionary of loss components
    #     """
    #     seg_logits = self.forward(inputs)
    #     losses = self.loss_by_feat(seg_logits, batch_data_samples)
    #     return losses
    
    def forward(self, inputs):
        output =  self.cls_seg(inputs)
        # output = inputs
        return output
    
    
    