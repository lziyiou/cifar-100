import torchvision

model = torchvision.models.vgg19().to('cuda')

# 修改最后一层以适应 CIFAR-100 数据集
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, 100)])  # Add new output layer
model.classifier = nn.Sequential(*features)

print(model.classifier)
