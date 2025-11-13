# Tsetlin Machine C
C library for Tsetlin Machines.
PLEASE RENAME REPO TO `tsetlin_machine_c` 🤐😣

## Features
- C library for Tsetlin Machines: inference, training, saving to / loading from bin files
- TM types: normal (dense), sparse, stateless (sparse)
- model import from green_tsetlin https://github.com/ooki/green_tsetlin

## TODOs
1. ~~Simplify project structure of Python and C files~~
2. ~~Add CMakeLists.txt for building and installing the project~~
3. ~~Add flatcc git submodules as dependencies~~
4. ~~Include unity test framework in CMakeLists.txt~~
5. Consider feasability of using memmap when loading models
6. Print learning debug (e.g. accuracy) every iteration during training
7. Add manually adding rules to the model
8. Add prediction explananations
9. Add README.md with project description, screenshots and usage instructions
10. Test exporting to edge devices with different architectures
11. (Optional) Complete and add FBS to Sparse TM
12. ~~Complete Python module for example notebooks and exporting models from green_tsetlin (`pip install .`)~~

## Requirements
- gcc
- make (optional)
- python (optional - importing models from green_tsetlin, demo data)
- uv (optional - if importing models from green_tsetlin, demo data)

## Install
- `git clone https://github.com/notEloiir/green_tsetlin_to_trainable_C.git`
- `cd green_tsetlin_to_trainable_C`
- `uv sync` (if using uv)

## Run demos
- MNIST training, data downloaded by python script
    - `uv run make run_mnist_demo`
    - `make run_mnist_demo_c` after first run
- MNIST inference using pretrained (dense) model, test data downloaded by python script
    - `uv run make run_mnist_inference_demo`
    - `make run_mnist_inference_demo_c` after first run
- Model size comparison of different TM types loading a pretrained (dense) model
    - `make run_model_size_demo`

## Run tests
- `make run_tests`
