import matplotlib.pyplot as plt
from os import listdir, getcwd
import pandas as pd


def get_y_at_x(x, xs, ys):
  
  # print(x, xs)
  if (x > max(xs) or x < min(xs)):
    return None
  low = 0
  high = len(xs) - 1
  closest_smaller = None
  while low <= high:
    mid = (low + high) // 2
    if xs[mid] < x:
      closest_smaller = mid
      low = mid+1
    elif xs[mid] > x:
      high = mid-1
    else:
      break
  
  # print("mid index: ", mid, " mid value : ", xs[mid])
  if mid >= len(xs):
    return None
  if xs[mid] == x:
    return ys[mid]
  
  if (xs[mid] > x):
    mid -= 1

  x1 = xs[mid]
  x2 = xs[mid+1]
  y1 = ys[mid]
  y2 = ys[mid+1]
  m = (y2 - y1) / (x2 - x1)

  interpolated_y = y1 + m * (x - x1)
  return interpolated_y


def create_avg_sequence_with_hetero_x(sequences):  
    
  xss, yss = sequences

  all_x = list(set([x for xs in xss for x in xs]))
  all_y = []
  all_x.sort()

  # print(all_x)
  for x in all_x:
    sum = 0
    misses = 0
    for xs, ys in zip(xss, yss):
      item = get_y_at_x(x, xs, ys)
      if item is None:
        misses += 1
      else:
        sum += item
    avg = sum / (len(xss) - misses)
    all_y += [avg]

  # print(all_x)
  # print(all_y)

  return all_x, all_y


def plot_training_data(index_columns, folders, plot_args=[]):

  # folders are structured as an iterable of strings,
  # each folder is loaded and the csvs within it are averaged
  # along the indicies taken. Typically the first index is 
  # just the epoch
  current_directory = getcwd()
  print("Current working directory:", current_directory)

  while len(plot_args) < len(folders):
    plot_args += [dict()]

  xc, yc = index_columns
  for folder, plargs in zip(folders, plot_args):
    csvs = [filename for filename in listdir(folder) if filename.endswith(".csv")]
    xs, ys = [], []
    for csvf in csvs:
      data = pd.read_csv(folder+csvf)

      xs += [data[xc]]
      ys += [data[yc]]

      # print(data.columns)
    x, y = create_avg_sequence_with_hetero_x((xs, ys))
    plt.plot(x,y,**plargs)
    # set label
  plt.legend(loc='upper center',  ncol=1)
  plt.show()
  