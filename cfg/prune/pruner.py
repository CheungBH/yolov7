import torch
import torch_pruning as tp
from models.yolo import Detect, IDetect
from models.common import ImplicitA, ImplicitM
import torch.nn as nn


class pruner:
    def __init__(self, model, device, opt):
        model.eval()
        example_inputs = torch.randn(1, 3, 640, 640).to(device)

        if opt.prune_norm == 'L2':
            imp = tp.importance.MagnitudeImportance(p=2)
        elif opt.prune_norm == 'L1':
            imp = tp.importance.MagnitudeImportance(p=1)
        elif opt.prune_norm == 'fpgm':
            imp = tp.importance.FPGMImportance(p=2)
        elif opt.prune_norm == 'lamp':
            imp = tp.importance.LAMPImportance(p=2)
        else:
            raise NotImplementedError("Pruning norm not implemented")

        ignored_layers = []
        for m in model.modules():
            if isinstance(m, (Detect, IDetect)):
                ignored_layers.append(m.m)
        unwrapped_parameters = []
        for m in model.modules():
            if isinstance(m, (ImplicitA, ImplicitM)):
                unwrapped_parameters.append((m.implicit, 1))  # pruning 1st dimension of implicit matrix

        iterative_steps = opt.epochs // opt.num_epochs_to_prune  # progressive pruning
        if opt.prune_method == "magnitude":
            sparsity_learning = False
            # if opt.prune_norm == 'L2':
            #     imp = tp.importance.MagnitudeImportance(p=2)
            # elif opt.prune_norm == 'L1':
            #     imp = tp.importance.MagnitudeImportance(p=1)
            # elif opt.prune_method == 'fpgm':
            #     imp = tp.importance.FPGMImportance(p=2)
            # elif opt.prune_method == 'lamp':
            #     imp = tp.importance.LAMPImportance(p=2)
            self.pruner = tp.pruner.MagnitudePruner(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=iterative_steps,
                ch_sparsity=opt.prun_ratio,
                ignored_layers=ignored_layers,
                global_pruning=True,
            )
        elif opt.prune_method == "bn_scale":
            sparsity_learning = True
            imp = tp.importance.BNScaleImportance()
            self.pruner = tp.pruner.BNScalePruner(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=iterative_steps,
                ch_sparsity=opt.prun_ratio,
                ignored_layers=ignored_layers,
                global_pruning=True,
            )
        elif opt.prune_method == "group_norm":
            sparsity_learning = False
            imp = tp.importance.GroupNormImportance(p=2)
            self.pruner = tp.pruner.GroupNormPruner(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=iterative_steps,
                ch_sparsity=opt.prun_ratio,
                ignored_layers=ignored_layers,
                global_pruning=True,
            )
        else:
            raise NotImplementedError("Pruning method not implemented")

        self.sparsity = opt.prun_ratio
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