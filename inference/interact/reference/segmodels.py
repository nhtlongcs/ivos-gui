from typing import Dict, Any
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

"""
Source: https://github.com/qubvel/segmentation_models.pytorch
"""

class BaseSegModel(nn.Module):
    """
    Some simple segmentation models with various pretrained backbones
    name: `str`
        model name [unet, deeplabv3, ...]
    encoder_name : `str` 
        backbone name [efficientnet, resnet, ...]
    num_classes: `int` 
        number of classes
    aux_params: `Dict` 
        auxilliary head
    """
    def __init__(
        self, 
        model_name: str, 
        encoder_name : str = "resnet34", 
        num_classes: int = 1000,
        aux_params: Dict = None,
        in_channels: int = 3,
        pretrained: bool = True,
        **kwargs):
        super().__init__()

        if pretrained:
            encoder_weights = "imagenet"
        else:
            encoder_weights = None

        self.num_classes = num_classes
        self.model = smp.create_model(
            arch = model_name,
            encoder_name = encoder_name,
            in_channels = in_channels,
            encoder_weights = encoder_weights,
            classes = num_classes, 
            aux_params = aux_params)

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward(self, batch: Dict, device: torch.device):
        x = batch['inputs'].to(device)
        outputs = self.model(x)
        return {
            'outputs': outputs,
        }

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.
        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 
        """
        outputs = self.forward(adict, device)['outputs']
        outputs = torch.softmax(outputs, dim=1) # B, C, H, W

        if 'weights' in adict.keys():
            weights = adict['weights'] # C
            for i, weight in enumerate(weights):
                outputs[:, i] *= weight

        predicts = outputs.detach().cpu().numpy()
        return {
            'masks': predicts
        } 