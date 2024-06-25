import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import cv2
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import torch
import torch.nn as nn

label_map = {'black_bear': 0, 'people': 1, 'birds': 2, 'dog': 3, 'brown_bear': 4, 'roe_deer': 5, 'wild_boar': 6, 'amur_tiger': 7, 'amur_leopard': 8, 'sika_deer': 9}
target_size = (224,224)

transforms = v2.Compose([
    v2.RandomResizedCrop(size=target_size, antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class RussianWildlifeDataset(Dataset):

    def __init__(self, img_dir):

        self.img_dir = img_dir
        self.data_list = []
        for label in os.listdir(self.img_dir):
            for file in os.listdir(os.path.join(self.img_dir,label)):
                self.data_list.append((file,label))

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, idx):

        file_name, label = self.data_list[idx]
        img_path = os.path.join(self.img_dir, label, file_name)

        img = torch.from_numpy(cv2.imread(img_path))
        img = img.permute(2, 0, 1)
        img = img/255
        img = transforms(img)
        return img.float(), label_map[label]
    
data = RussianWildlifeDataset('data')
print(len(data))

train_size = int(0.7 * len(data))
val_size = int(0.1 * len(data))
test_size = len(data) - train_size - val_size

train_set, val_set, test_set = random_split(data, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=8)

print(len(train_set), len(val_set), len(test_set))


class CNN(nn.Module):
  
  def __init__(self, num_classes):

    super(CNN, self).__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=4, stride=4)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.flatten = nn.Flatten()
    self.fc = nn.Linear(25088, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.fc(x)
    return x
  
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(10) # Assuming 10 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epoch = 10
save_loss = 99999

print("Started Training")

for epoch in range(n_epoch):
    print('Training for epoch: ', epoch)

    model.train()
    tloss = 0
    tstep = 0

    for i, data in enumerate(train_loader, 0):
        img, label = data
        inputs = img.to(device)
        labels = label.to(device)
        optimizer.zero_grad()

        inputs = inputs.squeeze(1)
        outputs = model(inputs)

        train_loss =  criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()
        tstep+=1
        tloss += train_loss.item()

    tstep+=1
    print('EPOCH:',epoch)
    print('Average train loss:', tloss/tstep)

    model.eval()
    vloss = 0
    vstep = 0

    for i, data in enumerate(val_loader, 0):
        img, label = data
        inputs = img.to(device)
        labels = label.to(device)

        inputs = inputs.squeeze(1)
        outputs = model(inputs)

        val_loss =  criterion(outputs, labels)
        vstep+=1
        vloss += val_loss.item()

    vstep+=1
    print('EPOCH:',epoch)
    print('Average val loss:', vloss/vstep)
    if(vloss/vstep<save_loss):
        save_loss = vloss/vstep
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, 'q2_cnn_aug_best.pt')  

    log_metric = {"Epoch":epoch, "Train Loss": tloss/tstep, "Val Loss": vloss/vstep}
    # wandb.log(log_metric)
        
print("Finished Training")

print("Started Testing")

test_labels = []
test_predictions = []

model.eval()

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        img, labels = data
        test_labels.extend(labels.numpy())

        inputs = img.to(device)
        inputs = inputs.squeeze(1)
        
        outputs = model(inputs).argmax(dim=1)
        test_predictions.extend(outputs.numpy())
        
print("Finished Testing")



test_labels = np.array(test_labels)
test_predictions = np.array(test_predictions)


correct = (test_labels==test_predictions).sum()
total = len(test_predictions)

acc = correct/total
f1score = f1_score(test_labels,test_predictions,average='macro')

print("Accuracy: ", acc)
print("F1 Score: ", f1score)

cm = confusion_matrix(test_labels, test_predictions)
# wandb.log({"Accuracy":acc, "F1 Score":f1score})
print(cm)

plt.imshow(cm, cmap='Blues')
plt.show(plt)
# wandb.log({"Confusion Matrix": wandb.Image(plt)})