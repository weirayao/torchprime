import torch.distributed as dist
import torch_xla.runtime as xr

xr.use_spmd()
dist.init_process_group(backend='gloo', init_method='xla://')

print(dist.get_world_size())