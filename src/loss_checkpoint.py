# Day 14: checkpoint on loss and perplexity, and a first check that PyTorch works.
# Run from the project root:  venv_gpt/bin/python src/loss_checkpoint.py

import math

import torch  # PyTorch: arrays that can also remember how they were made (like our Value)

# Loss and perplexity are two views of the same number: perplexity = e^loss.
# Losses below were measured on days 7, 12 and 13 on tiny Shakespeare (65 characters).
print(f"{'model':28} {'loss':>6} {'perplexity':>11}")
for name, loss in [
    ("perfect (always certain)", 0.0),
    ("bigram counts (day 12)", 2.4526),
    ("bigram trained (day 13)", 2.4639),
    ("blind guess over 65 chars", math.log(65)),
]:
    print(f"{name:28} {loss:6.3f} {math.exp(loss):11.1f}")  # :>6 right-aligns in 6 spaces

# Going back from perplexity to loss: log undoes exp.
assert abs(math.log(math.exp(2.4526)) - 2.4526) < 1e-12

# PyTorch check: a tensor is PyTorch's array. Same length of [3, 4] as day 3, expect 5.
t = torch.tensor([3.0, 4.0])
print("\ntorch version:", torch.__version__)
print("length of [3, 4]:", t.norm().item())  # .item() turns a one-number tensor into a Python number
assert t.norm().item() == 5.0
print("Apple GPU (mps) available:", torch.backends.mps.is_available())
