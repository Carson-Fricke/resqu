import copy
import functools
import re
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random as r
import matplotlib.pyplot as plt



class SingleVarFunctions(Dataset):

  # 0 for constants
  # 1 for variables (x)
  # 2 for open bracket
  # 3 for closed bracket
  # 4 for comma


  class FuncTree():

    class OperationNode():

      def __init__(self, inputs, operation, name):
        self.inputs = inputs
        self.operation = operation
        self.value = 0.0
        self.name = name

      def execute(self, x):
        pv = []
        for p in self.inputs:
          pv.append(p.execute(x))
        return self.operation(*pv)
      
      def __str__(self):
        if len(self.inputs) == 0:
          return self.name
        if len(self.inputs) == 1:
          if len(self.inputs[0].inputs) <= 1:
            return f'{self.name}({self.inputs[0]})'
          else:
            return f'{self.name}{self.inputs[0]}'
        elif len(self.inputs) == 2:
          return f'({self.inputs[0]} {self.name} {self.inputs[1]})'
        else:
          output = f'{self.name}('
          for inp in self.inputs[:-1]:
            output += f'{inp}, '
          output += f'{self.inputs[-1]})'
          return output
        
      def numerize(self):
        if len(self.inputs) == 1:
          if len(self.inputs[0].inputs) <= 1:
            # return f'{self.name}({self.inputs[0]})'
            t, v = self.inputs[0].numerize()
            return [self.name, 2, *t, 3], [0.0, 0.0, *v, 0.0]
          else:
            # return f'{self.name}{self.inputs[0]}'
            t, v = self.inputs[0].numerize()
            return [self.name, *t], [0.0, *v]
        elif len(self.inputs) == 2:
          t0, v0 = self.inputs[0].numerize()
          t1, v1 = self.inputs[1].numerize()
          # return f'({self.inputs[0]} {self.name} {self.inputs[1]})'
          return [2, *t0, self.name, *t1, 3], [0.0, *v0, 0.0, *v1, 0.0]
        else: 
          # output = f'{self.name}('
          ot, ov = [self.name, 2], [0.0, 0.0]
          for inp in self.inputs[:-1]:
            # output += f'{inp}, '
            t, v = inp.numerize()
            ot += [*t, 4]
            ov += [*v, 0.0]
          # output += f'{self.inputs[-1]})'
          t, v = self.inputs[-1].numerize()
          ot += [*t, 3]
          ov += [*v, 0.0]
          return ot, ov
      
    class ConstantNode(OperationNode):

      def __init__(self, value):
        super().__init__([], None, f'{value:.3f}')
        self.value = value

      def execute(self, x):
       return self.value
      
      def numerize(self):
        return [0], [float(self.value)]
      
    class VariableNode(OperationNode):

      def __init__(self):
        super().__init__([], None, 'x')

      def execute(self, x):
        return x
      
      def numerize(self):
        return [1], [0]
    
    def __init__(self, complexity, operations=[
      # (1, F.relu, 'relu'), 
      # (1, torch.cos, 'cos'), 
      (1, torch.sin, 'sin'),
      # (1, lambda x: torch.pow(x, 2), 'square'),
      # (1, lambda x: 1 / (x**2 + 1), 'f-gauss'),
      (2, torch.add, '+'), 
      (2, torch.mul, '*'), 
      (2, torch.sub, '-'), 
      # (2, lambda x,y: torch.pow(torch.abs(x), y), '^')
    ], constant_range=(-4,4)):
      self.complexity = complexity
      self.operations = operations
      nodes = [self.VariableNode()] + [self.VariableNode() if bool(r.getrandbits(1)) else self.ConstantNode(torch.tensor(r.uniform(*constant_range))) for _ in range(0,complexity)]

      op_stack = [x for x in operations for _ in range(complexity)]
      
      self.vocab_size = len(self.operations) + 5

      c_score = complexity
      while len(nodes) > 1 and c_score > 0:
        r.shuffle(nodes)
        r.shuffle(op_stack)
        op = op_stack.pop()
        inputs = [nodes.pop() for _ in range(op[0])]
        if len(inputs) == 1:
          pass
        nodes.append(self.OperationNode(inputs, op[1], op[2]))
        
      
      
      self.root = nodes[0]

    def __str__(self):
      return str(self.root)

    def execute(self, x):
      return self.root.execute(x)
    
    def plot(self, plot_range=(-3,3), n=1000):
      step = (plot_range[1] - plot_range[0]) / n
      x = torch.arange(start=plot_range[0], end=plot_range[1], step=step)
      y = self.execute(x)
      plt.plot(x.numpy(), y.numpy())
      plt.show()
    
    def generate_data(self, invalues):
      t, v = self.root.numerize()
      for opi, op in enumerate(self.operations, start=5):
        for i1, token in enumerate(t):
          if token == op[2]:
            t[i1] = opi

      tokens = torch.nn.functional.one_hot(torch.tensor(t), self.vocab_size)
      values = torch.tensor(v)
      outputs = self.execute(invalues)

      n = len(outputs)

      return tokens, values, invalues, outputs

    def is_nice(self, input_range=(-5,5), n=1000, magnitude_range=(0.1, 500), std_over_range_allowed=(0, 0.7)):
      step = (input_range[1] - input_range[0]) / n
      x = torch.arange(start=input_range[0], end=input_range[1], step=step)
      y = self.execute(x)
      mag = torch.max(torch.abs(y))

      std = torch.std(y)
      r = torch.max(y) - torch.min(y)
      stdr = std / r
      if mag > magnitude_range[1] or mag < magnitude_range[0]:
        return False
      if std < 0.1 or stdr > std_over_range_allowed[1]:
        return False
      return True

  def __init__(self, 
    num_funcs=2000, 
    samples=5000, 
    input_range=(-5, 5), 
    function_complexity_range=(2,7), 
    operations=[
      # (1, F.relu, 'relu'), 
      # (1, torch.cos, 'cos'), 
      (1, torch.sin, 'sin'), 
      # (1, lambda x: torch.pow(x, 2), 'square'),
      # (1, lambda x: 1 / (x**2 + 1), 'f-gauss'),
      # (1, lambda x: F.leaky_relu(x, negative_slope=0.1), 'leaky_relu'),
      # (1, lambda x: torch.pow(x, 1/3), 'cuberoot'),
      (2, torch.add, '+'), 
      (2, torch.mul, '*'), 
      (2, torch.sub, '-'), 
      # (2, torch.pow, '^')
    ], 
    seed=1,
    transform=None
  ):
    self.operations = operations
    r.seed(seed)
    self.funcs = []
    for _ in range(num_funcs):
      new_func = self.FuncTree(r.randint(*function_complexity_range), operations)
      while not new_func.is_nice(input_range=input_range):
        new_func = self.FuncTree(r.randint(*function_complexity_range), operations)
      self.funcs.append(new_func)

    step = (input_range[1] - input_range[0]) / samples
    self.x = torch.arange(start=input_range[0], end=input_range[1], step=step)

  def plot_funcs(self, pn=(10,10), plot_range=(-3,3), n=1000):
    step = (plot_range[1] - plot_range[0]) / n
    x = torch.arange(start=plot_range[0], end=plot_range[1], step=step)
    figure, axis = plt.subplots(*pn)
    for px in range(pn[0]):
      for py in range(pn[1]):
        y = self.funcs[py + px * pn[0]].execute(x)
        axis[px, py].plot(x,y)
    plt.show()

  def __len__(self):
    return len(self.funcs)

  @functools.cache
  def __getitem__(self, idx):
    return self.funcs[idx].generate_data(self.x)
