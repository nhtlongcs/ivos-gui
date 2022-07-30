import torch
import pickle
from inference.interact.reference.transunet.transunet_pos import TransUnetPE

class ReferenceController:
    def __init__(self, device = "cuda"):
        self.device = torch.device(device)
        self.model = TransUnetPE(
            model_name = "R50-ViT-B_16",
            img_size = 512 ,
            in_channels = 3,
            pretrained = False
        )

        self.model = self.model.to(self.device)

    def unanchor(self):
        self.anchored = False

    def interact(self, image, rel_pos):
        if isinstance(image, str):
            image_np = pickle.loads(image.encode("latin-1"))
            image = torch.from_numpy(image_np)
        # image = image.to(self.device, non_blocking=True)

        out_masks = self.model.get_prediction(
            {
                "inputs": image,
                "sids": [rel_pos]
            },
            self.device
        )["masks"]

        return out_masks
