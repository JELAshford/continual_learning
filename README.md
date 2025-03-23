# Continual Learning

Experiments in basic continual learning with small neural networks and the MNIST dataset. The main goal was demonstrating catastrohpic forgetting on a simple "2 classes at a time" continual learning task.

## Running

The project packages are managed with `uv`, and the main code can be setup and run with:

```bash
uv sync
uv run src/continual_learning/full_mnist_learning.py
```
