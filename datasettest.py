import torch
from src.synthetic_datasets.single_var_functions import SingleVarFunctions

dataset = SingleVarFunctions(seed=2)

dataset.plot_funcs()

print()
print(dataset.funcs[2])
print(dataset[1])