# Image Classification using PyTorch

I use google colab connected to google drive

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [DataLoader](#dataloader)
- [Method 1: Custom CNN Model](#method-1-custom-cnn-model)
- [Method 2: Transfer Learning with EfficientNet-B2](#method-2-transfer-learning-with-efficientnet-b2)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

This project demonstrates image classification using two approaches: building a custom CNN from scratch and utilizing transfer learning with a pre-trained EfficientNet-B2 model. The goal is to classify images accurately, with the second method outperforming the first and achieving the highest score in a Kaggle competition.

## Dataset

Create class dataset in pytorch to transform image and label each class. I split dir_image and set take number similar to index.
transform to augmented dataset

```sh
    class ModiaDataset(Dataset):
    def __init__(self, images, csv_file,mode = "train" ,transform = None):
        self.images = images
        self.labels = pd.read_csv(csv_file)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        if self.mode == "train" or self.mode == "val":
            img = Image.open(image_name)
            label = self.labels[self.labels.index == int(image_name.split("/")[-1].split(".")[0])]["label"].values
            label = torch.tensor(label, dtype = torch.long)
            img = self.transform(img)
            return img, label - 1

        elif self.mode == "test":
            img = Image.open(image_name)
            img = self.transform(img)
            return img, image_name
```

## Method 1: Custom CNN Model

Create model CNN to train. Downsampling each layer into 4 times and increase channels into 2 times. I use softmax to classify 4 classes. I also used hyperparameter tuning but the accuracy quite not good so I continue to method 2

```sh
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size = 7, stride = 2, padding = 0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 5, stride = 2, padding = 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(2048, 1000)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1000, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out
```

## Method 2: Transfer Learning with EfficientNet-B2

In this I use pre-trained model, epecially is the EfficientNet-B2 algorithm. I also use transfer learning with freezing layer and change fully connected layers

```sh
from torchvision import models
model = models.efficientnet_b2(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Linear(num_features, 1000, bias = True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 4, bias = True),
)

```

## Results

Method 2, which uses transfer learning with the EfficientNet-B2 model, achieved the highest accuracy and outperformed the custom CNN model. This method also secured the highest score in a Kaggle competition.

## Conclusion

This project demonstrates the effectiveness of transfer learning in image classification tasks. By fine-tuning a pre-trained EfficientNet-B2 model, we achieved superior performance compared to a custom CNN built from scratch.
