import pytest
import os, sys
sys.path.append(os.getcwd())
sys.path.append('..')
import torch as tc
from bptt import plrnn
import main

def test_plrnn_initializes_with_default_args_without_dataset():
    args = main.get_default_args()
    model = plrnn.PLRNN(args)

def test_plrnn_equations_with_fixed_parameters():
    z = tc.tensor([[1.0,-1.0]])
    A = tc.tensor([0.5, 0.5])
    W1 = tc.zeros((2, 3))
    W2 = tc.zeros((3, 2))
    h1 = tc.zeros(2)
    h2 = tc.zeros(3)
    z1 = plrnn.PLRNN.PLRNN_step('shallow-PLRNN', z, A, W1, W2, h1, h2)
    assert (z1 == tc.tensor([[0.5, -0.5]])).all()

if __name__=='__main__':
    test_plrnn_equations_with_fixed_parameters()