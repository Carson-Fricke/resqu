import matplotlib.pyplot as plt
from os import listdir
import pandas as pd


def plot_training_data(index_columns, folders):

  # folders are structured as an iterable of strings,
  # each folder is loaded and the csvs within it are averaged
  # along the indicies taken. Typically the first index is 
  # just the epoch
  xc, yc = index_columns
  for folder in folders:
    csvs = [filename for filename in listdir(folder) if filename.endswith(".csv")]
    for csvf in csvs:
      data = pd.read_csv(csvf)
      print(data.columns)


  plt.plot()
  