import torch
from src.synthetic_datasets.single_var_functions import SingleVarFunctions

dataset = SingleVarFunctions()

# dataset.plot_funcs()

print(dataset.funcs[2])
print(dataset[1])