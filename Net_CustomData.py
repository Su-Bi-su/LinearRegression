import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Custom dataset
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])


# Build the model
class LinReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linreg = nn.Linear(1,1)

    def forward(self, x):
        return self.linreg(x)


# Training the model

# Move everything(data and model both)to GPU if available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LinReg().to(device)
x = x.to(device)
y = y.to(device)

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.000001)
lossfn = nn.MSELoss(size_average=False)

training_loss = []
# Train the model
for epoch in range(4000):
    #model.train()
    optimizer.zero_grad()               # it basically null in each iteration

    y_hat = model(x)
    loss = lossfn(y_hat, y)
    print(epoch, loss.data.item())
    training_loss.append(loss.data.item())

    # Back-propagation

    loss.backward(loss)
    optimizer.step()

fig, ax = plt.subplots()
ax.plot(range(4000), training_loss, ".")
ax.set_title("Loss Curve")
plt.xlabel('#epoches')
plt.ylabel('Loss')
plt.show()

# Eval
print("\nAfter Training : ")
x_new = torch.tensor([[20.0]])
x_new = x_new.to(device)
with torch.no_grad():
    y_hat_new = model(x_new)
    print(x_new.item(), "------->", y_hat_new.item())






