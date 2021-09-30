"""Module to create a custom :mod:`DetrR50 <alonet.detr.detr_r50>` model which allows to upload a decided pretrained
weights and change the `class_embed` layer in order to train custom classes.
"""

from alonet.detr import DetrR50
from torch import nn

from argparse import Namespace


class DetrR50Finetune(DetrR50):
    """Pre made helpfull class to finetune the :mod:`DetrR50 <alonet.detr.detr_r50>` model on a custom class.

    Parameters
    ----------
    num_classes : int
        Number of classes to use
    background_class : int, optional
        Background class id, by default the last id
    *args : Namespace
        Arguments used in :mod:`DetrR50 <alonet.detr.detr_r50>` module
    **kwargs : dict
        Aditional arguments used in :mod:`DetrR50 <alonet.detr.detr_r50>` module
    """

    def __init__(self, num_classes: int, background_class: int = None, *args: Namespace, **kwargs: dict):
        """Init method"""
        super().__init__(*args, background_class=background_class, **kwargs)
        self.num_classes = num_classes
        # Replace the class_embed layer a new layer once the detr-r50 weight are loaded
        # + 1 to include the background class.
        self.background_class = self.num_classes if background_class is None else background_class
        self.num_classes = num_classes + 1
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)
        self.class_embed = self.class_embed.to(self.device)


if __name__ == "__main__":
    # Setup a new Detr Model with 2 class and the background class equal to 0.
    # Additionally, we're gonna load the pretrained detr-r50 weights.
    detr_r50_finetune = DetrR50Finetune(num_classes=2, weights="detr-r50")
