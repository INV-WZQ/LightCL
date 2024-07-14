import torch


def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)
    num_elements = tensor.numel()
    num_zeros = round(tensor.numel()*sparsity)
    importance = torch.abs(tensor)
    threshold = torch.kthvalue(importance.flatten(), num_zeros)
    mask = importance > threshold[0]
    return mask

class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1 and name in sparsity_dict: # we only prune conv and fc weights
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks

def get_num_channel_to_keep(channels: int, prune_ratio: float) -> int:
    return int(channels * (1. - prune_ratio)+0.5)


def channel_prune(param, dict_sparsity):
    (x, y, hx, hy) = param.size()
    now = param.detach().mean(dim=-1).mean(-1).mean(-1)
    n_keep = get_num_kernel_to_keep(x , dict_sparsity)
    threshold = torch.kthvalue(now.flatten(), max(1,x - n_keep))
    mask = now >= threshold[0]
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    mask_repeated = mask_expanded.repeat(1, y, hx, hy)
    return mask_repeated

class ChannelPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = ChannelPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        last = None
        for name, param in model.named_parameters():
            if ('linear' in name) or ('classifier' in name):
                if 'weight' in name:
                    masks[name] = last.repeat(param.size()[0], 1)
                continue
            if param.dim() > 1 and name in sparsity_dict:
                tmp = None
                if last!=None and 'shortcut' not in name:
                    tmp = last.unsqueeze(1).unsqueeze(1)
                    tmp = tmp.expand(last.size()[0], 3,3)
                if sparsity_dict[name] == 0.0:
                    masks[name] = torch.ones_like(param)
                else: 
                    masks[name] = channel_prune(param, sparsity_dict[name])
                if tmp != None:
                    tmp = tmp.expand_as(masks[name])
                    masks[name] = masks[name]*tmp
                last = masks[name].detach()
                last = last.view(last.size()[0], -1).max(dim=1).values
            else:
                masks[name] = last
        return masks

def get_num_kernel_to_keep(kernel: int, prune_ratio: float) -> int:
    return int(kernel * (1. - prune_ratio)+0.5)

def kernel_prune(model, dict_sparsity): 
    (x, y, hx, hy) = param.size()
    now = param.detach().mean(dim=-1).mean(-1)
    n_keep = get_num_kernel_to_keep(x * y, dict_sparsity[name])
    threshold = torch.kthvalue(now.flatten(), max(1,x * y - n_keep))
    mask = now >= threshold[0]
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
    mask_repeated = mask_expanded.repeat(1, 1, hx, hy)
    #param.mul_(mask_repeated)
    return mask_repeated


class KernelPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = ChannelPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]
                
    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if ('linear' in name) or ('classifier' in name):continue
            if param.dim() > 1 and name in sparsity_dict: # we only prune conv and fc weights
                masks[name] = kernel_prune(param, sparsity_dict[name])
        return masks
