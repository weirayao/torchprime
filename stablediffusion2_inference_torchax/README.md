# Install dependencies
1. torch & Torchxla2

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/pytorch/xla.git
cd xla/experimental/torch_xla2
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e .[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

2. diffusers

```
pip install diffusers
```

# How to run:

```
python sdxl.py
```

NOTE:
This version uses the stablediffusion model defined in 
huggingface diffusers package. The next step would be 
to show case a fork and changes to make it more performant
