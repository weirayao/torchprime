# Llama training



## Features

- Optimized for PyTorch/XLA
- Demonstrates GSPMD parallelism
- Supports large language models tasks

## Requirements

- PyTorch
- PyTorch/XLA
- Transformers library
- Datasets library

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/pytorch-tpu/TorchPrime.git
   cd TorchPrime/llama_training_xla
   ```

2. Install the package:
   ```
   pip install -e .
   ```

3. Run the training script:
   ```
   XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python trainer.py configs/run.json
   ```

## Key Components

- `trainer.py`: Main training script that sets up the model, data, and training loop
- `configs/run.json`: Configuration file for the training script
- `configs/fsdp_config.json`: Configuration file for FSDPv2
- `torchprime/models/llama.py`: Implementation of the Llama model

## Contributing

Contributions to TorchPrime are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the New BSD License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- PyTorch team for the amazing deep learning framework
- Google's XLA team for the accelerated linear algebra compiler
- Hugging Face for the Transformers library

For more information on PyTorch/XLA, visit the [official documentation](https://github.com/pytorch/xla).
