import torch
import torch.nn as nn

# from .datasets import *
from .flows import (
    MADE,
    BatchNormFlow,
    CouplingLayer,
    InvertibleMM,
    LUInvertibleMM,
    MADESplit,
    Reverse,
    FlowSequential,
)


def get_flow(
    flow_name, num_inputs, num_hidden, num_cond_inputs, num_blocks, act="relu",
):
    modules = []
    assert flow_name in ["maf", "maf-split", "maf-split-glow", "realnvp", "glow"]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if flow_name == "glow":
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        print("Warning: Results for GLOW are not as good as for MAF yet.")
        for _ in range(num_blocks):
            modules += [
                BatchNormFlow(num_inputs),
                LUInvertibleMM(num_inputs),
                CouplingLayer(
                    num_inputs,
                    num_hidden,
                    mask,
                    num_cond_inputs,
                    s_act="tanh",
                    t_act="relu",
                ),
            ]
            mask = 1 - mask
    elif flow_name == "realnvp":
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        for _ in range(num_blocks):
            modules += [
                CouplingLayer(
                    num_inputs,
                    num_hidden,
                    mask,
                    num_cond_inputs,
                    s_act="tanh",
                    t_act="relu",
                ),
                BatchNormFlow(num_inputs),
            ]
            mask = 1 - mask
    elif flow_name == "maf":
        for _ in range(num_blocks):
            modules += [
                MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
                BatchNormFlow(num_inputs),
                Reverse(num_inputs),
            ]
    elif flow_name == "maf-split":
        for _ in range(num_blocks):
            modules += [
                MADESplit(
                    num_inputs, num_hidden, num_cond_inputs, s_act="tanh", t_act="relu"
                ),
                BatchNormFlow(num_inputs),
                Reverse(num_inputs),
            ]
    elif flow_name == "maf-split-glow":
        for _ in range(num_blocks):
            modules += [
                MADESplit(
                    num_inputs, num_hidden, num_cond_inputs, s_act="tanh", t_act="relu"
                ),
                BatchNormFlow(num_inputs),
                InvertibleMM(num_inputs),
            ]
    return modules