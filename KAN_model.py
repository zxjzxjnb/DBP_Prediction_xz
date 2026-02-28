import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from kan import *
from kan.utils import create_dataset
from kan.utils import ex_round

torch.manual_seed(0)
np.random.seed(0)

df = pd.read_csv('data.csv')
df = df.drop('Sample', axis=1)
data = np.array(df)
data = np.float32(data)

data = torch.from_numpy(data)
# print(data.dtype)
x = data[:, 5:]
y = data[:, 0:5]
# y = torch.unsqueeze(y, dim=1)
# print(f'Input data shape: {x.shape}')
# print(f'Output data shape: {y.shape}')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):

        ID = self.x[index]

        # Load data and get label
        X = self.x[index]
        Y = self.y[index]

        return X, Y

# create a KAN: 8D inputs, 5D output, and 3 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[8, 4, 5], grid=8, k=5, seed=42)


# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=8)

dataset['train_input'] = x
dataset['train_label'] = y

dataset['test_input'] = x[40:]
dataset['test_label'] = y[40:]

# print(dataset['train_input'].shape, dataset['train_label'].shape)


# print(model)
# plot KAN at initialization
model(dataset['train_input'])
# model.plot(beta=100, title = 'My KAN', metric='act')
# plt.show()


model.fit(dataset, opt="LBFGS", steps=100, lamb=0)
train_prediction = model(dataset['train_input'])
print(torch.sum((dataset['train_label']-train_prediction)**2))

test_prediction = model(dataset['test_input'])
print(torch.sum((dataset['test_label']-test_prediction)**2))

model = model.prune()
model.fit(dataset, opt="LBFGS", steps=100, lr=0.01)
test_prediction = model(dataset['test_input'])
print(torch.sum((dataset['test_label']-test_prediction)**2))

# train_prediction = model(dataset['test_input'])
# print(torch.sum((dataset['test_label']-train_prediction)**2))
# model = model.refine(10)
# model.fit(dataset, opt="LBFGS", steps=20000, lr=0.01)
# train_prediction = model(dataset['test_input'])
# print(torch.sum((dataset['test_label']-train_prediction)**2))
mode = "auto" # "manual"

# if mode == "manual":
#     # manual mode
#     model.fix_symbolic(0,0,0,'sin')
#     model.fix_symbolic(0,1,0,'x^2')
#     model.fix_symbolic(1,0,0,'exp')
# elif mode == "auto":
#     # automatic mode
#     lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
#     model.auto_symbolic(lib=lib)

model.plot()
plt.show()
print(model.suggest_symbolic(0,0,0, weight_simple=0.0))
print(model.suggest_symbolic(0,1,0, weight_simple=0.0))
print(model.suggest_symbolic(0,2,0, weight_simple=0.0))
print(model.suggest_symbolic(0,3,0, weight_simple=0.0))
print(model.suggest_symbolic(0,4,0, weight_simple=0.0))

print(model.suggest_symbolic(1,0,0, weight_simple=0.0))
print(model.suggest_symbolic(1,1,0, weight_simple=0.0))
print(model.suggest_symbolic(1,2,0, weight_simple=0.0))
print(model.suggest_symbolic(1,3,0, weight_simple=0.0))

model.fit(dataset, opt="LBFGS", steps=400, lamb=0, lr=0.1)

print()
print(ex_round(model.symbolic_formula()[0][0], 4))
print()
print(ex_round(model.symbolic_formula()[0][1], 4))
print()
print(ex_round(model.symbolic_formula()[0][2], 4))
print()
print(ex_round(model.symbolic_formula()[0][3], 4))
print()
print(ex_round(model.symbolic_formula()[0][4], 4))

test_prediction = model(dataset['test_input'])
print(torch.sum((dataset['test_label']-test_prediction)**2))

# model.plot()
# plt.show()
