#!/usr/bin/env python
# coding: utf-8

# In[79]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:





# In[80]:


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[81]:


# Define transformations for the train and test sets
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[115]:


data_dir = 'F:\\Study Materials\\4-1\\Thesis\\Mulberry dataset'
all_data = datasets.ImageFolder(data_dir, transform=train_transforms)

train_data = datasets.ImageFolder('H:/Research/plant disease/Data/Train/fold_1')
test_data = datasets.ImageFolder('H:/Research/plant disease/Data/test/fold_1')
val_data = datasets.ImageFolder('H:/Research/plant disease/Data/val/fold_1')

# Define data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


# In[116]:


len(train_data)


# In[113]:


train_data[0]


# In[117]:


# Load pre-trained VGG19 model
base_model = models.vgg19(pretrained=True)
for param in base_model.parameters():
    param.requires_grad = False


# In[118]:


# Add additional layers on top of the base model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.base_model = base_model
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# In[119]:


model = CNN(num_classes=38).to(device)


# In[120]:


from torchsummary import summary

summary(model,(3,224,224))


# In[121]:


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# In[122]:


# Train the model
num_epochs = 3
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    val_loss = 0
    val_correct = 0
    
    # Train the model on the training set
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accs.append(train_accuracy)
    
    # Test the model on the validation set
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100. * val_correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

    # Print training and validation statistics
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy))


# In[126]:


train_data = datasets.ImageFolder('H:\\Research\\plant disease\\Data\\Train\\fold_1')
val_data = datasets.ImageFolder('H:\\Research\\plant disease\\Data\\Val\\fold_1')
test_data = datasets.ImageFolder('H:\\Research\\plant disease\\Data\\Test\\fold_1')

# Define data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


# In[ ]:





# In[146]:


# train_data = datasets.ImageFolder('H:\\Research\\plant disease\\Data\\Train\\fold_1')
# val_data = datasets.ImageFolder('H:\\Research\\plant disease\\Data\\Val\\fold_1')
# test_data = datasets.ImageFolder('H:\\Research\\plant disease\\Data\\Test\\fold_1')

# # Define data loaders
# batch_size = 32
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
# # Train the model
# num_epochs = 10
# train_losses = []
# val_losses = []
# train_accs = []
# val_accs = []

# for epoch in range(num_epochs):
#     train_loss = 0
#     train_correct = 0
#     val_loss = 0
#     val_correct = 0
    
#     # Train the model on the training set
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         pred = output.argmax(dim=1, keepdim=True)
#         train_correct += pred.eq(target.view_as(pred)).sum().item()
#     train_loss /= len(train_loader.dataset)
#     train_accuracy = 100. * train_correct / len(train_loader.dataset)
#     train_losses.append(train_loss)
#     train_accs.append(train_accuracy)
    
#     # Test the model on the validation set
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(val_loader):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)
#             val_loss += loss.item()
#             pred = output.argmax(dim=1, keepdim=True)
#             val_correct += pred.eq(target.view_as(pred)).sum().item()
#         val_loss /= len(val_loader.dataset)
#         val_accuracy = 100. * val_correct / len(val_loader.dataset)
#         val_losses.append(val_loss)
#         val_accs.append(val_accuracy)

#     # Print training and validation statistics
#     print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
#           .format(epoch+1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy))


# In[147]:


train_dir='H:\\Research\\plant disease\\Data\\Train\\fold_1'
test_dir='H:\\Research\\plant disease\\Data\\Test\\fold_1'
val_dir='H:\\Research\\plant disease\\Data\\Val\\fold_1'


# In[153]:


import torch
from torchvision import datasets, transforms

# Define custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = datasets.ImageFolder(root_dir, transform=transform)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# Define transforms to convert PIL Images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
    
])

# Define custom datasets
train_data = CustomDataset(train_dir,transform=transform)
val_data = CustomDataset(test_dir,transform=transform)
test_data = CustomDataset(val_dir,transform=transform)

# Define data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


# In[154]:


len(val_loader)


# In[156]:



# Train the model
num_epochs = 20
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    val_loss = 0
    val_correct = 0
    
    # Train the model on the training set
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        # Calculate and print the percentage of current epoch progress
        progress = (batch_idx / len(train_loader)) * 100
        print('\rEpoch: {} [{}/{} ({:.0f}%)]'.format(epoch+1, batch_idx * len(data), len(train_loader.dataset), progress), end='')
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accs.append(train_accuracy)
    
    # Test the model on the validation set
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()
            progress = (batch_idx / len(val_loader)) * 100
            print('\rEpoch: {} [{}/{} ({:.0f}%)]'.format(epoch+1, batch_idx * len(data), len(val_loader.dataset), progress), end='')
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100. * val_correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

    # Print training and validation statistics
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy))


# In[ ]:


# Plot training and validation accuracies
plt.plot(train_accs, label='Training accuracy')
plt.plot(val_accs, label='Validation accuracy')
plt.title('Training and Validation Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()


# In[107]:


print(train_accuracy)
print(val_accuracy)


# In[ ]:


# Save the trained model
#torch.save(model.state_dict(), 'plant_disease_detection_vgg19.pth')

