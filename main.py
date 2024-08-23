import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torchvision import datasets

tfm = v2.Compose([
    v2.RandomResizedCrop(size=(32, 32), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

tfm_val = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

batch_size = 256
num_workers = 6
epochs = 10
lr = 1e-3

train_ds = datasets.CIFAR100(root='./data', train=True, download=False, transform=tfm)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_ds = datasets.CIFAR100(root='./data', train=False, download=False, transform=tfm_val)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 64, 1024),
            nn.PReLU(1024),
            nn.Linear(1024, 100)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = Net().to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for images, labels in tqdm(train_dl, desc=f'Epoch {epoch+1}'):
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    for images, labels in tqdm(val_dl):
        images, labels = images.to('cuda'), labels.to('cuda')
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {correct / len(val_ds)}')
    """
    Accuracy: 0.3643
    """
