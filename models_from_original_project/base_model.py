import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

df = pd.read_csv('data.csv')
df = df.drop('Sample', axis=1)
data = np.array(df)
data = np.float32(data)

# BUG: No feature scaling. UVA254 ~ 0.01 vs NH4-N ~ 30 → gradient dominated by large-scale features.
# Fix: apply StandardScaler before converting to tensor.
data = torch.from_numpy(data)
# print(data.dtype)
x = data[:, 5:]
y = data[:, :5]
# y = torch.unsqueeze(y, dim=1)
print(f'Input data shape: {x.shape}')
print(f'Output data shape: {y.shape}')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):

        ID = self.x[index]  # BUG: unused variable, redundant with X below.

        # Load data and get label
        X = self.x[index]
        Y = self.y[index]

        return X, Y

# BUG: Fixed sequential split without shuffle or cross-validation → unreliable evaluation.
train_dataset = Dataset(x[:40], y[:40])
vali_dataset = Dataset(x[40:], y[40:])
# print(dataset[5])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
# BUG: batch_size=100 > len(vali_dataset)=26; also shuffle=True is unnecessary for validation.
vali_loader = torch.utils.data.DataLoader(dataset=vali_dataset, batch_size=100, shuffle=True)

# a, b = next(iter(train_loader))
print(f'Training samples: {len(train_dataset)}')
print(f'Validation samples: {len(vali_dataset)}')


# BUG: 2130 params vs 40 train samples (53x overparameterized). No Dropout/BN → guaranteed overfitting.
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()#继承
        self.hidden_1 = torch.nn.Linear(n_feature,n_hidden)   # hidden layer
        self.hidden_2 = torch.nn.Linear(n_hidden,100)
        self.hidden_3 = torch.nn.Linear(100,n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        # print(x.shape)
        x1 = self.hidden_1(x)
        x2 = F.relu(x1)
        x3 = self.hidden_2(x2)
        x4 = F.relu(x3)
        x5 = self.hidden_3(x4)
        x6 = F.relu(x5)
        y = self.predict(x6)
        return y

net = Net(n_feature=x.shape[1], n_hidden=10, n_output=y.shape[1])     # define the network
# print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# plt.ion()   # something about plotting

# BUG: 1M iters with no early stopping, no best-model checkpoint, no LR schedule.
for t in range(1000000):
    net.train()
    # BUG: iter() recreates the iterator each call → always fetches a random batch instead of
    # iterating through all batches. Fix: use nested loop `for x_, y_ in train_loader`.
    x_, y_ = next(iter(train_loader))
    prediction = net(x_)     # input x and predict based on x

    # print(prediction.shape, y.shape)
    loss = loss_func(prediction, y_)     # must be (1. nn output, 2. target)
    # print(loss)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5000 == 0:
        net.eval()
        x_val, y_val = next(iter(vali_loader))
        prediction = net(x_val)     # input x and predict based on x
        loss_val = loss_func(prediction, y_val)     # must be (1. nn output, 2. target)
        # BUG: prints train `loss` instead of `loss_val`. Fix: replace `loss` → `loss_val`.
        print(f'Epoch {t}: Validation loss: {loss}')
#         # plot and show learning process
#         plt.cla()
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#         plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
#         plt.pause(0.1)
# print(prediction)
# plt.ioff()
# plt.show()
# for i in range(len(y)):
#     print(prediction[i], y[i])

# BUG: No final evaluation (MSE/R² per target), no model saving → results not reproducible.
