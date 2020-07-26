from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


n_samples = 100
n_features = 1   #input_dimension


# data loading (Extract)
x, y = make_regression(n_features=n_features, n_samples=n_samples, noise=10)

#fig, ax = plt.subplots()

#ax.plot(x, y, '.')

#plt.show()

# transform to tensor(Transform)
x = torch.from_numpy(x).float()
y = torch.from_numpy(y.reshape((n_samples, n_features))).float()

# Loading

# Build the model
class LinReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linreg = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x):
        return self.linreg(x)


# Training the model

# Move everything(data and model both)to GPU if available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LinReg(n_features, 1).to(device)
x = x.to(device)
y = y.to(device)

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.00001)
lossfn = nn.MSELoss()

# Train the model
for epoch in range(100):
    model.train()
    optimizer.zero_grad()               # it basically null in each iteration

    y_hat = model(x)
    loss = lossfn(y_hat, y)

    # Back-propagation

    loss.backward(loss)
    optimizer.step()

# Eval
model.eval()
with torch.no_grad():
    y_hat = model(x)


# Visualize
fig, ax = plt.subplots()
ax.plot(x.cpu().numpy(), y_hat.cpu().numpy(), ".", label='pred')
ax.plot(x.cpu().numpy(), y.cpu().numpy(), ".", label='data')
ax.set_title(f"MSE : {loss.item(): 0.1f}")
ax.legend()
plt.show()


