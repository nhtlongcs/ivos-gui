import torch
import pickle
from inference.interact.reference.transunet.transunet_pos import TransUnetPE
from inference.interact.reference.cps import CrossPseudoSupervision
from inference.interact.reference.segmodels import BaseSegModel

class ReferenceController:
    def __init__(self, checkpoint_path=None, device = "cuda"):
        self.device = torch.device(device)
        model1 = TransUnetPE(
            model_name = "R50-ViT-B_16",
            img_size = 512 ,
            in_channels = 3,
            pretrained = False,
            num_classes=14
        )

        model2 = BaseSegModel(
            model_name='deeplabv3plus',
            encoder_name="efficientnet-b3",
            in_channels=3,
            num_classes=14,
            pretrained=False
        )

        self.model = CrossPseudoSupervision(
            model1,
            model2,
            reduction="sum"
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            model1_state_dict = state_dict['model1']
            model2_state_dict = state_dict['model2']

            self.model.model1.load_state_dict(model1_state_dict)
            self.model.model2.model.load_state_dict(model2_state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

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
