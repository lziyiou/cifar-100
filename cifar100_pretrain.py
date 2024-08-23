import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
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
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = datasets.CIFAR100(root='./data', train=True, download=False, transform=tfm)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_ds = datasets.CIFAR100(root='./data', train=False, download=False, transform=tfm_val)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


if __name__ == '__main__':
    weights = torchvision.models.VGG19_Weights.DEFAULT
    model = torchvision.models.vgg19(weights=weights, progress=True).to(device)

    # 修改最后一层以适应 CIFAR-100 数据集
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 100)])  # Add new output layer
    model.classifier = nn.Sequential(*features)

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

        # 每个 epoch 结束后评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch + 1}/{epochs}], Accuracy: {correct / total}')
        model.train()  # 将模型切换回训练模式

    model.eval()
    correct = 0
    for images, labels in tqdm(val_dl):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {correct / len(val_ds)}')
    """
    Accuracy: 
    """
