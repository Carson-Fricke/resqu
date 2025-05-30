# import torch
# from src.synthetic_datasets.single_var_functions import SingleVarFunctions
from src.visualize_data.graph import *
# dataset = SingleVarFunctions(seed=2)
# 
# dataset.plot_funcs()
# 
# print()
# print(dataset.funcs[2])
# print(dataset[1])

# nx, ny = create_avg_sequence_with_hetero_x(([[1,2,3,4], [1,2,2.5,3,4]], [[4,3,2,1], [4.2,3.2,2.7,2.2,1.2]]))
# 
# plt.plot(nx, ny)
# plt.plot([1,2,3,4], [4,3,2,1])
# plt.plot([1,2,3,4], [4.2,3.2,2.2,1.2])
# plt.show()

# print(get_y_at_x(0.9, [1,2,3,4], [1,2,3,10]))


plot_training_data(
  ('epoch', 'train_loss'),      
  [
    './linear-autoencoder-mnist/rate0.000/', 
    './linear-autoencoder-mnist/rate0.062/', 
    './linear-autoencoder-mnist/rate0.125/',
    './linear-autoencoder-mnist/rate0.250/',
    './linear-autoencoder-mnist/rate0.500/'
  ],
  [ 
    {"label": "rr0.000"},
    {"label": "rr0.062"},
    {"label": "rr0.125"},
    {"label": "rr0.250"},
    {"label": "rr0.500"}
  ]
)
