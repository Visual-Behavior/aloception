# Inspired by the official DETR repository and adapted for aloception
# https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/detr.py
"""
Panoptic module to predict object segmentation.
"""
import os

import torch
import torch.nn.functional as F
from torch import nn

from alonet.detr.detr_r50 import DetrR50
from alonet.panoptic.nn import FPNstyleCNN
from alonet.panoptic.nn import MHAttentionMap
from alonet.detr.misc import assert_and_export_onnx

import aloscene
import alonet


class PanopticHead(nn.Module):
    """Head Pytorch module to predict segmentation masks from previous boxes detection task.

    Parameters
    ----------
    DETR_module : alonet.detr.Detr
        Object detection module based on DETR architecture
    aux_loss: bool, optional
        Return aux outputs in forward step (if required), by default use DETR_module.aux_loss attribute value
    device : torch.device, optional
        Configure module in CPU or GPU, by default torch.device("cpu")
    """

    INPUT_MEAN_STD = alonet.detr.detr.INPUT_MEAN_STD

    def __init__(
        self,
        DETR_module: alonet.detr.Detr,
        aux_loss: bool = None,
        device: torch.device = torch.device("cpu"),
        weights: str = None,
    ):
        super().__init__()
        self.detr = DETR_module

        # Get complement outputs in forward
        self.detr.return_dec_outputs = True
        self.detr.return_enc_outputs = True
        self.detr.return_bb_outputs = True
        self.detr.aux_loss = aux_loss or self.detr.aux_loss

        # Freeze DETR parameters to not train them
        for p in self.parameters():
            p.requires_grad_(False)

        # Use values provides of DETR module in MHA and FPN
        hidden_dim, nheads = DETR_module.transformer.d_model, DETR_module.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = FPNstyleCNN(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

        if device is not None:
            self.to(device)

        self.device = device

        # Load weights
        if weights is not None:
            weights = os.path.join(alonet.common.weights.vb_fodler(), "weights", weights, weights + ".pth")
            alonet.common.load_weights(self, weights, device, strict_load_weights=True)

    @assert_and_export_onnx(check_mean_std=True, input_mean_std=INPUT_MEAN_STD)
    def forward(self, frames: aloscene.frame, **kwargs):
        """PanopticHead forward, that joint to the previous boxes predictions the new masks feature.

        Parameters
        ----------
        frames : aloscene.frame
            Input frame to network

        Returns
        -------
        Dict
            It outputs a dict with the following elements:
                - "pred_logits": The classification logits (including no-object) for all queries.
                                Shape = [batch_size x num_queries x (num_classes + 1)]
                - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                                (center_x, center_y, height, width). These values are normalized in [0, 1],
                                relative to the size of each individual image (disregarding possible padding).
                                See PostProcess for information on how to retrieve the unnormalized bounding box.
                - "pred_masks": Binary masks, each one to assign to predicted boxes.
                                Shape = [batch_size x num_queries x H // 4 x W // 4]
                - "bb_outpus": Backbone outputs, requered in this forward
                - "enc_outpus": Transformer encoder outputs, requered on this forward
                - "dec_outpus": Transformer decoder outputs, requered on this forward
                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # DETR model forward to obtain box embeddings
        out = self.detr(frames, **kwargs)
        features = out["bb_outputs"]

        src, mask = features[-1][0], features[-1][1].to(torch.float32)
        mask = mask[:, 0].to(torch.bool)
        bs = src.shape[0]

        # FIXME h_boxes takes the last one computed, keep this in mind
        # Use box embeddings as input of Multi Head attention
        bbox_mask = self.bbox_attention(out["dec_outputs"][-1], out["enc_outputs"], mask=mask)
        # And then, use MHA ouput as input of FPN-style CNN
        seg_masks = self.mask_head(self.detr.input_proj(src), bbox_mask, [features[i][0] for i in range(3)[::-1]])

        out["pred_masks"] = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        out["frame_size"] = frames.shape[-2:]
        return out  # Return the DETR output + pred_masks

    @torch.no_grad()
    def inference(self, forward_out: dict, filterth=0.2, maskth=0.5, **kwargs):
        """Given the model forward outputs, this method will return an aloscene.Mask tensor with binary
        mask with its corresponding label per object detected.

        Parameters
        ----------
        forward_out: dict
            Dict with the model forward outptus

        Returns
        -------
        aloscene.BoundingBoxes2D
            Boxes from DETR model
        aloscene.Maks
            Binary mask of PanopticHead
        """

        # Get probs to labels select from detr model
        outs_logits = forward_out["pred_logits"]
        outs_probs = F.softmax(outs_logits, -1)
        outs_scores, outs_labels = outs_probs.max(-1)
        filters = self.detr.get_outs_filter(
            outs_scores=outs_scores, outs_labels=outs_labels, threshold=filterth, **kwargs,
        )

        # Process each feature map
        outputs_masks = forward_out["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(
            outputs_masks, size=forward_out["frame_size"], mode="bilinear", align_corners=False
        )
        outputs_masks = (outputs_masks.sigmoid() > maskth).cpu()

        preds_masks = []
        for scores, labels, cur_mask, b_filter in zip(outs_scores, outs_labels, outputs_masks, filters):
            scores, labels = scores[b_filter], labels[b_filter]
            labels = aloscene.Labels(labels.type(torch.float32), encoding="id", scores=scores, names=("N",))
            masks = cur_mask[b_filter]
            # masks = F.interpolate(masks.float(), size=tuple(tt.tolist()), mode="nearest").byte()
            masks = aloscene.Mask(masks, names=("N", "H", "W"), labels=labels)
            preds_masks.append(masks)

        return self.detr.inference(forward_out), preds_masks


if __name__ == "__main__":
    device = torch.device("cuda")

    # Load model
    model = DetrR50(num_classes=91, weights="detr-r50", device=device).eval()
    model = PanopticHead(model)
    model.to(device)

    # Random frame
    # frame = aloscene.Frame(torch.rand((3, 250, 250)), names=("C", "H", "W")).norm_resnet().to(device)
    frame = aloscene.Frame("./images/aloception.png", names=("C", "H", "W")).norm_resnet().to(device)
    frame = frame.resize((300, 300))
    frame = frame.batch_list([frame])
    frame.get_view().render()

    # Pred of size (B, NQ, H//4, W//4)
    print(model(frame)["pred_masks"].shape)
    pred = model.inference(model(frame))
    print("size of each pred:", [len(p) for p in pred])

    frame.get_view([pred[0].get_view(frame[0])]).render()
