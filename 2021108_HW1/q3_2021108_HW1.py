import os
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import gc

class IndianDrivingDataset(Dataset):

    def __init__(self, img_dir, label_dir):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_list = []
        self.label_list = []
        for i in os.listdir(self.img_dir):
            temp = i.split('_')
            self.img_list.append('image_archive/image_'+temp[1])
            self.label_list.append('mask_archive/mask_'+temp[1])
        self.centre_crop = transforms.CenterCrop((512,512))

    def __len__(self):

        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):

        img_path= self.img_list[idx]
        label_path = self.label_list[idx]
        img = Image.open(img_path)
        img_tensor = torch.from_numpy(np.array(img))
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.type(torch.float32) / 255.0
        img_tensor = self.centre_crop(img_tensor)
        label = Image.open(label_path)
        label = np.array(label)
        label_tensor = torch.from_numpy(np.array(label))
        label_tensor = self.centre_crop(label_tensor)
        return img_tensor, label_tensor
    
data = IndianDrivingDataset('image_archive','mask_archive')
print(len(data))

train_size = int(0.7 * len(data))
test_size = len(data) - train_size

train_set, test_set = random_split(data, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=2, shuffle=False, num_workers=8)

print(len(train_set),len(test_set))

counter=100

pixel_count_unique = {}
for i in range(26):
    pixel_count_unique[i] = 0
pixel_count_unique[255] = 0


for image, label in train_loader:
    counter-=2
    if(counter==0):
        break
    temp = label.numpy()
    temp = np.unique(label[0])
    for i in temp:
        pixel_count_unique[i]+=1
    del image, temp
    temp = np.unique(label[1])
    for i in temp:
        pixel_count_unique[i]+=1
    del temp, label
    gc.collect()

print(pixel_count_unique)

classes = ('road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'person', 'rider', 'motorcycle',
'bicycle', 'auto-rickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail',
'billboard', 'traffic sign', 'traffic light', 'pole', 'obs fallback', 'building', 'bridge', 'vegetation', 'sky',
'unlabelled')

temp_list = [i for i in range(26)]
temp_list.append(255)

label_map = {}
for i, j in enumerate(classes):
    if(i==26):
        break
    label_map[j] = pixel_count_unique[i]
label_map['unlabelled'] = pixel_count_unique[255]
print(label_map)

fig = plt.figure(figsize = (20, 5))
plt.bar(list(label_map.keys()), list(label_map.values()))

img, label = data[0]

plt.figure(figsize=(16,12))

plt.subplot(1,2,1)
plt.imshow(img.permute((1, 2, 0))) 
plt.axis('off')

plt.subplot(1,2,2)
im = plt.imshow(label*31)
plt.axis('off')

plt.show()

import network

model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=8)
model.load_state_dict( torch.load('best_deeplabv3plus_mobilenet_cityscapes_os16.pth', map_location='cpu')['model_state'])

id_dict = {'road':0, 'sidewalk':1, 'building':2, 'wall':3, 'fence':4, 'pole':5, 'traffic light':6, 'traffic sign':7, 'vegetation':8, 'terrain':9, 'sky':10, 'person':11, 'rider':12, 'car':13,
           'truck':14, 'bus':15, 'train':16, 'motorcycle':17, 'bicycle':18, 'unlabelled': 255}

class_dict = {}

for i in id_dict:
    class_dict[id_dict[i]] = i
# this on what mobilenet model was trained on.

converted_dict = {0:0, 1:0, 2:1, 3:9, 4:11, 5:12, 6:17, 7:18, 8:13, 9:13, 10:14, 11:15, 12:15, 13:3, 14:3, 15:4, 16:4, 17:7, 18:7, 19:6, 20:5, 21:5, 22:2, 23:2, 24:8, 25:10, 255:255}

def calculate_mAP(seg_gt, seg_pred, num_classes=19):
    list_precision = []
    list_recall = []

    avg_precision = 0.0
    for class_id in range(num_classes):
        class_gt = (seg_gt == class_id).astype(np.float32)
        class_pred = (seg_pred == class_id).astype(np.float32)

        true_positives = np.sum(class_gt * class_pred)
        false_positives = np.sum(class_pred) - true_positives
        false_negatives = np.sum(class_gt) - true_positives

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        list_precision.append(precision)
        list_recall.append(recall)

        avg_precision += precision

    mAP = avg_precision / num_classes
    return mAP, list_precision, list_recall

def calculate_mIoU(seg_gt, seg_pred, num_classes=19):
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for class_id in range(num_classes):
        class_gt = (seg_gt == class_id).astype(np.float32)
        class_pred = (seg_pred == class_id).astype(np.float32)

        intersection[class_id] = np.sum(class_gt * class_pred)
        union[class_id] = np.sum(class_gt) + np.sum(class_pred) - intersection[class_id]

    IoU = intersection / (union + 1e-8)
    mIoU = np.mean(IoU)
    return IoU

num_counter = 0
pixel_accuracy = 0
dice_coeff = 0
total_mAP = 0

total_mIoU = {}
total_precision = {}
total_recall = {}
for i in range(19):
    total_mIoU[i] = (0,0)
    total_precision[i] = (0,0)
    total_recall[i] = (0,0)


def visualise(pred, label):
    plt.figure(figsize=(16,12))

    plt.subplot(1,2,1)
    im = plt.imshow(pred)
    plt.axis('off')

    plt.subplot(1,2,2)
    im = plt.imshow(label)
    plt.axis('off')

    plt.show()

def metric_func(pred, label, flag=False):
    global num_counter, pixel_accuracy, dice_coeff, total_mIoU, total_mAP, total_precision, total_recall
    num_counter+=1
    for i in range(512):
        for j in range(512):
            label[i,j] = converted_dict[label[i,j]]

    pixel_wise_accuracy = np.mean(pred == label)
    intersection = np.sum(pred * label)
    union = np.sum(pred) + np.sum(label)
    dice_coefficient = (2.0 * intersection) / union
    mAP, precision, recall = calculate_mAP(label,pred)
    mIoU = calculate_mIoU(label,pred)
    if(flag):
        print(mIoU)
    if(flag):
        visualise(pred, label)
    for i, j in enumerate(mIoU):
        if j!=0:
            a, b = total_mIoU[i]
            a+=j
            b+=1
            total_mIoU[i] = (a,b)
    pixel_accuracy += pixel_wise_accuracy
    dice_coeff += dice_coefficient
    total_mAP += mAP
    for i, j in enumerate(precision):
        if j!=0:
            a, b = total_precision[i]
            a+=j
            b+=1
            total_precision[i] = (a,b)
    for i, j in enumerate(recall):
        if j!=0:
            a, b = total_recall[i]
            a+=j
            b+=1
            total_recall[i] = (a,b)

y_pred = []
y_true = []

for image, label in test_loader:
    flag = False
    if(num_counter>=100):
        break
    with torch.no_grad():
        outputs = model(image)
    preds = outputs.max(1)[1].detach().numpy()
    labels = label.numpy()
    if(num_counter<6):
        flag = True
    metric_func(preds[0],labels[0],flag)
    metric_func(preds[1],labels[1],flag)
    y_pred.append(preds[0])
    y_true.append(labels[0])
    y_pred.append(preds[1])
    y_true.append(labels[1])
    del image, labels, outputs
    gc.collect()

print('Performance')

print('Pixel Wise Accuracy: ', pixel_accuracy/num_counter*100)
print('Dice Coefficient Accuracy: ', dice_coeff/num_counter)
print('mAP', total_mAP/num_counter*100)

print('Performance Class Wise for 100 images')
for i in range(19):
    if(i!=19):
        print(class_dict[i])
    else:
        print('Unlabelled')
    a, b = total_mIoU[i]
    if(b==0):
        print("mIoU:",'nan')
    else:
        print("mIoU:", a/b*100)

    a, b = total_recall[i]
    if(b==0):
        print("Recall:",'nan')
    else:
        print("Recall:", a/b*100)

    a, b = total_precision[i]
    if(b==0):
        print("Precision:",'nan')
    else:
        print("Precision:", a/b*100)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.figure(figsize=(16,12))
y_true = np.array(y_true).flatten()
y_pred = np.array(y_pred).flatten()
cm = confusion_matrix(y_true, y_pred, labels=range(19))
plt.imshow(cm, cmap='viridis')
plt.show(plt)