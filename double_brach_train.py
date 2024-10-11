import os
from torchvision import transforms, datasets
import torch
import cv2
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader

import pandas as p
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# 导入 AGGT 类
from ShuffleTransormer import ShuffleTransformer  # 假设 AGGT 类定义在名为 AGGT.py 的文件中


class MyDataset(Dataset):
    def __init__(self, origin_root, filter_root, transform, step='train'):
        self.origin_root = origin_root
        self.filter_root = filter_root
        self.images = []
        self.transform = transform
        self.labels = {}
        self.step = step
        images_path = glob.glob(os.path.join(self.origin_root, self.step, '*/*.JPG'))
        for index_path in images_path:
            folder_path, file_name = os.path.split(index_path)
            folders = folder_path.split('/')
            special_path = folders[-1]+'/'+file_name
            self.images.append(special_path)
            label = 0 if folders[-1] == 'fake' else 1
            self.labels[special_path] = label

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path1 = os.path.join(self.origin_root, self.step, img_name)
        img_path2 = os.path.join(self.filter_root, self.step, img_name)
        pil_image1 = Image.open(img_path1).convert("RGB")
        pil_image2 = Image.open(img_path2).convert("RGB")
        if self.transform:
            data1 = self.transform(pil_image1)
            data2 = self.transform(pil_image2)
        else:
            data1 = torch.from_numpy(pil_image1)
            data2 = torch.from_numpy(pil_image2)
        label = self.labels[img_name]
        return data1, data2, label

    def __len__(self):
        return len(self.images)
    

class AGGT50DualInput(nn.Module):
    def __init__(self, num_classes=2):
        super(AGGT50DualInput, self).__init__()
        self.features1 = ShuffleTransformer()
        self.features2 = ShuffleTransformer()
        self.fc = nn.Linear(1000 * 2, num_classes)  # 2048是AGGT50的输出特征数，*2是因为有两个支路

    def forward(self, x1, x2):
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x = torch.cat((x1, x2), dim=1)  # 在特征维度上连接两个支路的输出
        x = self.fc(x)
        return x

origin_root = '/home/liubn/0-liubaoning(136)/dataset/origin'
filter_root = '/home/liubn/0-liubaoning(136)/dataset/filter'
data_transform = {
    "train": transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize((224, 224)),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_dataset = MyDataset(origin_root, filter_root, data_transform["train"], 'train')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

test_dataset = MyDataset(origin_root, filter_root, data_transform["test"], 'test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

val_dataset = MyDataset(origin_root, filter_root, data_transform["test"], 'test')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

# 创建 AGGT 模型，根据需要调整输入通道数和输出类别数
num_classes = 2  # 二分类
model = AGGT50DualInput()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
model.to(device)
# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

best_val_accuracy = 0.0
best_model_path = '/home/liubn/0-liubaoning(136)/logs/aggt_best_model.pth'

def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data1, data2, labels in val_loader:
            outputs = model(data1.to(device), data2.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100):
    global best_val_accuracy
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data1, data2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data1.to(device), data2.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation after each epoch
        val_accuracy = validate_model(model, val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Loss: {running_loss / len(train_loader)} "
              f"Validation Accuracy: {val_accuracy:.2f}%")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved!")

# 在训练模型时调用 train_model 函数，并传入验证集的 DataLoader
train_model(model, criterion, optimizer, train_loader, val_loader)

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data1, data2, labels in test_loader:
            outputs = model(data1.to(device), data2.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

# 计算混淆矩阵
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 测试模型
true_labels, predicted_labels = test_model(model, test_loader)

# 绘制混淆矩阵
class_names = ['fake', 'real']
plot_confusion_matrix(true_labels, predicted_labels, class_names)
