import numpy as np
import torch
from thop import profile
from torchprofile import profile_macs
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis


def getParamCount(model, printLayers=False):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1 and printLayers:
                print(name+':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            elif printLayers:
                print(name+':', num_param)
            total_param += num_param
    
    print("\nTotal Trainable Parameters:", total_param)
    return total_param


def getFlops(model, x, device='cuda', debugging=False, profiler='torchprofile'):
    # https://github.com/sovrasov/flops-counter.pytorch/issues/16
    # https://github.com/Lyken17/pytorch-OpCounter
    # https://github.com/sovrasov/flops-counter.pytorch
    # https://github.com/zhijian-liu/torchprofile
    model.eval()

    with torch.no_grad():
        # x = torch.rand(1, 3, image_size, image_size).to(device)
        if profiler == 'torchprofile':
            flops = profile_macs(model, x)
        elif profiler == 'ptflops':
            flops, _ = get_model_complexity_info(
                model,  x.shape[1:], as_strings=False,
                print_per_layer_stat=debugging, verbose=debugging)
        elif profiler == 'fvcore':
            flops = FlopCountAnalysis(model, x)
            flops = flops.total()
        else:
            flops, _ = profile(model, inputs=(x, ))

    return flops
    
