import time
import torch
import torch_pruning as tp
from models.yolo import Detect, IDetect, IKeypoint
from models.common import ImplicitA, ImplicitM
import torch.nn as nn


class pruner:
    def __init__(self, model, device, opt):
        model.eval()
        example_inputs = torch.randn(1, 3, 640, 640).to(device)
        imp = tp.importance.MagnitudeImportance(p=2 if opt.prune_norm == 'L2' else 1)  # L2 norm pruning

        ignored_layers = []
        for m in model.modules():
            if isinstance(m, (Detect, IDetect, IKeypoint)):
                ignored_layers.append(m.m)
            elif isinstance(m, nn.Conv2d) and m.out_channels == 153:
                ignored_layers.append(m)
        unwrapped_parameters = []
        for m in model.modules():
            if isinstance(m, (ImplicitA, ImplicitM)):
                unwrapped_parameters.append((m.implicit, 1))  # pruning 1st dimension of implicit matrix

        iterative_steps = opt.epochs // opt.num_epochs_to_prune  # progressive pruning
        if opt.prune_method == "magnitude":
            self.pruner = tp.pruner.MagnitudePruner(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=iterative_steps,
                ch_sparsity=opt.sparsity,
                ignored_layers=ignored_layers,
                global_pruning=True,
                unwrapped_parameters=unwrapped_parameters
            )
        elif opt.prune_method == "bn_scale":
            self.pruner = tp.pruner.BNScalePruner(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=iterative_steps,
                ch_sparsity=opt.sparsity,
                ignored_layers=ignored_layers,
                global_pruning=True,
                unwrapped_parameters=unwrapped_parameters
            )
        else:
            raise NotImplementedError("Pruning method not implemented")
        # self.pruner = tp.pruner.MagnitudePruner(
        #     model,
        #     example_inputs,
        #     importance=imp,
        #     iterative_steps=iterative_steps,
        #     ch_sparsity=opt.sparsity,
        #     ignored_layers=ignored_layers,
        #     unwrapped_parameters=unwrapped_parameters
        # )
        self.sparsity = opt.sparsity
        self.num_steps = iterative_steps
        self.count = 0

    def step(self, model, device):
        self.count += 1

        example_inputs = torch.randn(1, 3, 640, 640).to(device)
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

        self.pruner.step()

        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print("Pruning Sparsity=%f" % (self.sparsity / self.num_steps * self.count))
        print("Before Pruning: MACs=%f, #Params=%f" % (base_macs, base_nparams))
        print("After Pruning: MACs=%f, #Params=%f" % (pruned_macs, pruned_nparams))