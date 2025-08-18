import torch
import torch_xla.core.xla_model as xm
dev = xm.xla_device()
x = torch.randn(32,32, device=dev)
y = x @ x.t()
xm.mark_step()
print("OK: matmul")