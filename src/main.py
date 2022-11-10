import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


# -----------------
# MODEL
# -----------------
class MNISTClassifier(nn.Module):

    def __init__(self):
        super(MNISTClassifier, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x


# ----------------
# DATA
# ----------------
data_directory = os.path.join(os.getcwd(), 'data')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = MNIST(data_directory, train=True, download=True, transform=transform)
mnist_test = MNIST(data_directory, train=False, download=True, transform=transform)

# train (55,000 images), val split (5,000 images)
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
mnist_test = MNIST(data_directory, train=False, download=True)

# The dataloaders handle shuffling, batching, etc...
mnist_train = DataLoader(mnist_train, batch_size=64)
mnist_val = DataLoader(mnist_val, batch_size=64)
mnist_test = DataLoader(mnist_test, batch_size=64)

# ----------------
# OPTIMIZER
# ----------------
pytorch_model = MNISTClassifier()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=1e-3)


# ----------------
# LOSS
# ----------------
def cross_entropy_loss(logits, labels):
    return F.nll_loss(logits, labels)


# ----------------
# TRAINING LOOP
# ----------------
num_epochs = 10
for epoch in range(num_epochs):

    train_running_loss = 0.0
    train_running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    # TRAINING LOOP
    for train_batch in mnist_train:
        x, y = train_batch

        logits = pytorch_model(x)
        loss = cross_entropy_loss(logits, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        _, preds = torch.max(logits, 1)
        train_running_loss += loss.item()
        train_running_corrects += torch.sum(preds == y.data)

    # VALIDATION LOOP
    with torch.no_grad():

        for val_batch in mnist_val:
            x, y = val_batch
            logits = pytorch_model(x)
            loss = cross_entropy_loss(logits, y)

            _, preds = torch.max(logits, 1)
            val_running_loss += loss.item()
            val_running_corrects += torch.sum(preds == y.data)

    epoch_train_loss = train_running_loss / len(mnist_train.dataset)
    epoch_train_accuracy = train_running_corrects.float() / len(mnist_train.dataset)
    epoch_val_loss = val_running_loss / len(mnist_val.dataset)
    epoch_val_accuracy = val_running_corrects.float() / len(mnist_val.dataset)

    # Print epoch statistics
    print(
        f"epoch: {epoch + 1:02d}/{num_epochs}    "
        f"training loss: {epoch_train_loss:.4f}    "
        f"training accuracy {epoch_train_accuracy.item():.4f}    "
        f"validation loss: {epoch_val_loss:.4f}    "
        f"validation accuracy: {epoch_val_accuracy.item():.4f}",
        flush=True,
    )
