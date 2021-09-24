from torch.utils.data import SequentialSampler
import numpy as np
import argparse
import torch
import os

from alonet.common.pl_helpers import load_training
from alodataset import SintelFlowDataset, Split
from alonet.raft.utils import Padder
from alonet.raft import LitRAFT
from alonet import ALONET_ROOT
from aloscene import Frame

from alonet.raft.trt.timing import load_trt_model
from alonet.raft.trt.timing import load_torch_model


def sintel_transform_fn(frame):
    return frame["left"].norm_minmax_sym()


# def parse_args():
#     parser = argparse.ArgumentParser(description="Raft evaluation on Sintel training set")
#     parser.add_argument("--weights", default="raft-things", help="name or path to weights file")
#     parser.add_argument(
#         "--project_run_id", default="raft", help="project run_id (if loading weights from a previous training)"
#     )
#     parser.add_argument("--run_id", help="load weights from previous training with this run_id")
#     args = parser.parse_args()
#     if args.run_id is not None:
#         args.weights = None
#     return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["torch", "trt"])
    return parser.parse_args()


def load_dataset():
    return SintelFlowDataset(
        cameras=["left"],
        labels=["flow"],
        passes=["clean"],
        sequence_size=2,
        transform_fn=sintel_transform_fn,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="backend")
    torch_parser = subparsers.add_parser("torch")
    trt_parser = subparsers.add_parser("trt")
    trt_parser.add_argument("precision", choices=["fp16", "fp32", "mix"])
    trt_parser.add_argument("--iters", type=int, default=12)
    trt_parser.add_argument("--engine_path")
    return parser.parse_args()


if __name__ == "__main__":

    kwargs = vars(parse_args())
    backend = kwargs.pop("backend")

    if backend == "trt":
        precision = kwargs["precision"]
        iters = kwargs["iters"]
        engine_path = os.path.join(ALONET_ROOT, f"raft-things_iters{iters}_{precision}.engine")
        model = load_trt_model(engine_path)
    else:
        model = load_torch_model()
    dataset = load_dataset()

    padder = Padder()  # pads inputs to multiple of 8

    with torch.no_grad():
        epe_list = []
        for idx, frames in enumerate(dataset.train_loader(batch_size=1, sampler=SequentialSampler)):
            if idx % 10 == 0:
                print(f"{idx+1:04d}/{len(dataset)}", end="\r")

            frames = Frame.batch_list(frames)
            frames = frames[..., 0:368, 0:496]
            frame1 = frames[:, 0, ...].clone()
            frame2 = frames[:, 1, ...].clone()
            # frame1 = padder.pad(frame1)
            # frame2 = padder.pad(frame2)

            if backend == "trt":
                frame1 = frame1.as_tensor().numpy()
                frame2 = frame2.as_tensor().numpy()
                # if precision == "fp16":
                #     frame1 = frame1.astype(np.float16)
                #     frame2 = frame2.astype(np.float16)
                flow_pred = model(frame1, frame2)["flow_up"]
            else:
                frame1 = frame1.to("cuda")
                frame2 = frame2.to("cuda")
                flow_pred = model(frame1, frame2, iters=12, only_last=True)["flow_up"]
                flow_pred = flow_pred.cpu()
                # flow_pred = padder.unpad(flow_pred)[0].numpy()
            flow_gt = frames[0][0].flow["flow_forward"].as_tensor().numpy()
            epe = np.sqrt(((flow_pred - flow_gt) ** 2).sum(axis=0))
            epe_list.append(epe.flatten())

        epe = np.mean(np.concatenate(epe_list))
        print("\n\nValidation Sintel EPE: %f" % epe)