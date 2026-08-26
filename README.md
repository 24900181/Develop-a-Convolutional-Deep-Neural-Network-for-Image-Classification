# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model

<img width="998" height="698" alt="image" src="https://github.com/user-attachments/assets/a757df16-cd3e-4a0a-99c3-8ac7e249f2af" />

## DESIGN STEPS
### STEP 1:  Collect and preprocess the image dataset.



### STEP 2: Import required deep learning libraries.



### STEP 3: Build the CNN architecture.



### STEP 4: Train the CNN model using training data.



### STEP 5: Evaluate the model performance using test data.



### STEP 6: Test the model with new images and verify predictions.





## PROGRAM

### Name: NETHRA.K

### Register Number: 212224230184

```python
!pip install torchsummary
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from torch.utils.data import DataLoader

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
train_set=torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,transform=transform)
test_set=torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,transform=transform)
im,lbl=train_set[0]
print(im.shape)

print(len(train_set))
print(len(test_set))
trl=DataLoader(train_set,batch_size=32,shuffle=True)
tstl=DataLoader(test_set,batch_size=32,shuffle=False)
class CNNclassifier(nn.Module):
    def __init__ (self):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.l1=nn.Linear(128*3*3,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,10)

    def forward(self,x):
        x=self.pool(torch.relu(self.c1(x)))
        x=self.pool(torch.relu(self.c2(x)))
        x=self.pool(torch.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=self.l3(x)
        return x





model=CNNclassifier()
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
summary(model,input_size=(1,28,28))

epochs=3
rl=0.0
for i in range(epochs):
    op.zero_grad()
    for im,lbl in trl:
        pred=model(im)
        loss=criterion(pred,lbl)
        loss.backward()
        op.step()
        rl+=loss.item()
    print(f"Running Loss:{i+1}/{epochs}",rl)



act=[]
pre=[]
crt=0
total=0
with torch.no_grad():
    for im,lbl in tstl:
        output=model(im)
        _,pred=torch.max(output,1)
        crt+=(pred==lbl).sum().item()
        total+=lbl.size(0)
        pre.extend(pred.numpy())
        act.extend(lbl.numpy())

accuracy=crt/total*100
print("Accuracy: ",accuracy)

        
     





```

### OUTPUT

## Training Loss per Epoch


<img width="872" height="171" alt="image" src="https://github.com/user-attachments/assets/0cb3f31d-c276-4f8a-b9c3-98e433659cd1" />


## Confusion Matrix


<img width="1110" height="840" alt="image" src="https://github.com/user-attachments/assets/121c5ecc-f686-4d3f-ab7e-1cd400d579bf" />


## Classification Report

<img width="971" height="525" alt="image" src="https://github.com/user-attachments/assets/c1a46f72-6e1c-4870-aebc-ab7d38b597e8" />


### New Sample Data Prediction

<img width="687" height="667" alt="image" src="https://github.com/user-attachments/assets/2f28ecfb-c58c-4893-9e77-cadd5519310d" />



<img width="321" height="108" alt="image" src="https://github.com/user-attachments/assets/c2f15bf3-5b5e-4101-9751-aca28de747da" />







## RESULT
The CNN model was successfully trained and tested, achieving accurate image classification for new input images.
