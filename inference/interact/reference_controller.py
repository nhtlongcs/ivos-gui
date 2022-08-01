import torch
import pickle
from inference.interact.reference.transunet.transunet_pos import TransUnetPE
import numpy as np

class ReferenceController:
    def __init__(self, checkpoint_path=None, device = "cuda"):
        self.device = torch.device(device)
        self.model = TransUnetPE(
            model_name = "R50-ViT-B_16",
            img_size = 512 ,
            in_channels = 3,
            pretrained = False,
            num_classes=14
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)['model']
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

    def unanchor(self):
        self.anchored = False

    def interact(self, image, rel_pos):
        if isinstance(image, str):
            image_np = pickle.loads(image.encode("latin-1"))
            image = torch.from_numpy(image_np)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            out_masks = self.model.get_prediction(
                {
                    "inputs": image,
                    "sids": [rel_pos]
                },
                self.device
            )["masks"]

        return out_masks
