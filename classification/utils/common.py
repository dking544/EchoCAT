from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch import distributed as dist


def is_module_wrapper(module):


    module_wrappers = tuple([DataParallel, DistributedDataParallel])
    return isinstance(module, module_wrappers)

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
