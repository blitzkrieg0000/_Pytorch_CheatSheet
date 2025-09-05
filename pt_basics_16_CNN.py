"""
    !=> Hedef: 
        => Pytorch ile CNN(Convolutional Neural Network) çalışması
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# CUDA CHECK
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True



#! PARAMS--------------------------------------------------------------------------------------------------------------
EPOCHS = 5
batch_size = 4
LEARNING_RATE = 0.001
SAVED_MODEL_PATH = "weight/model.pth"


# Veriseti PILImage tipinde resimler olan [0, 1] aralığında normalleştirilmiştir.. 
# Resimleri [-1, 1] aralığına normalize edip, Tensor tipine de çeviriyoruz.
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



#! PREPARE DATASET-----------------------------------------------------------------------------------------------------
# CIFAR10 Veriseti: 60000 toplam veriseti, 32x32 boyutlu görseller, 10 sınıf, her sınıfta 6000 görsel olan bir verisetidir.
train_dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def ShowImage(img):
    img = img / 2 + 0.5  # unnormalize (veriyi eski haline çevir)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


## Get Random Image & Show Image 
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# ShowImage(torchvision.utils.make_grid(images))



#! Create Model--------------------------------------------------------------------------------------------------------
# mxm CNN katmanı çıkış boyutu = ((W-F + 2*P)/S +1)
# W: image size
# F: filter size
# P: padding size
# S: stride size
class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.Pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = nn.Conv2d(6, 16, 5)
        self.FullyConnected1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.FullyConnected2 = nn.Linear(120, 84)
        self.FullyConnected3 = nn.Linear(84, 10)


    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.Pool(F.relu(self.Conv1(x)))  # -> n, 6, 14, 14
        x = self.Pool(F.relu(self.Conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400        # Flattening(Fully Connected Neural Network için 2D den 1D boyuta indirgiyoruz.)
        x = F.relu(self.FullyConnected1(x))   # -> n, 120
        x = F.relu(self.FullyConnected2(x))   # -> n, 84
        x = self.FullyConnected3(x)           # -> n, 10

        return x


model = MyConvNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

#! Train---------------------------------------------------------------------------------------------------------------
n_total_steps = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward & Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

print("Eğitim Bitti.")


#! Save Model----------------------------------------------------------------------------------------------------------
torch.save(model.state_dict(), SAVED_MODEL_PATH)


#! Test Model----------------------------------------------------------------------------------------------------------
# Modelin doğru tahmin etme oranını hesapla
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1


    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc} %")

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]}: {acc} %")




