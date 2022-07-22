import argparse
from unittest import mock

# This test use cuda. It should be ignore by github actions.

def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


# DETR TEST

from alonet.detr.train_on_coco import main as detr_train_on_coco, get_arg_parser as detr_get_arg_parser

detr_args = get_argparse_defaults(detr_get_arg_parser())
detr_args["weights"] = "detr-r50"
detr_args["train_on_val"] = True
detr_args["fast_dev_run"] = True
detr_args["sample"] = True


@mock.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(**detr_args))
def test_detr(mock_args):
    detr_train_on_coco()


# DeforambleDETR TEST

from alonet.deformable_detr.train_on_coco import (
    main as def_detr_train_on_coco,
    get_arg_parser as def_detr_get_arg_parser,
)

def_detr_args = get_argparse_defaults(def_detr_get_arg_parser())
def_detr_args["weights"] = "deformable-detr-r50"
def_detr_args["model_name"] = "deformable-detr-r50"
def_detr_args["train_on_val"] = True
def_detr_args["fast_dev_run"] = True
def_detr_args["sample"] = True


@mock.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(**def_detr_args))
def test_deformable_detr(mock_args):
    def_detr_train_on_coco()


# RAFT TEST

from alonet.raft.train_on_chairs import main as raft_train_on_chairs, get_args_parser as raft_get_args_parser

raft_args = get_argparse_defaults(raft_get_args_parser())
raft_args["weights"] = "raft-things"
raft_args["train_on_val"] = True
raft_args["fast_dev_run"] = True
raft_args["sample"] = True


@mock.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(**raft_args))
def test_raft(mock_args):
    raft_train_on_chairs()


if __name__ == "__main__":
    test_deformable_detr()
    test_detr()
    test_raft()
