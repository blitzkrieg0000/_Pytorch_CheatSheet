"""
    !=> Hedef: 
        => Pytorch ile Temel bir Neural Network nasıl eğitilir.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from lib.tool import CalculateConfusionMatrix


# CUDA CHECK
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch'u NVIDIA CUDA desteği ile kullanabiliyorsunuz.") if torch.cuda.is_available() else print("PyTorch'u NVIDIA CUDA desteği ile KULLANAMIYORSUNUZ. Bu yüzden CPU ile çalışmak yavaş olacaktır.")

# CUDA PERFORMANCE
torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


# PARAMETERS
INPUT_SIZE = 784    # 28x28 MNIST Dataset
HIDDEN_SIZE = 100
CLASSES = 10        # MNIST Dataset Digits: 0-9
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3



#! PREPARE DATASET-----------------------------------------------------------------------------------------------------
# MNIST DATASET
train_dataset = torchvision.datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


## Show and Exit
ShowSample = False
if ShowSample:
    examples = iter(train_loader)
    samples, labels = next(examples)
    for i in range(6):
        plt.subplot(2, 3, i+1) 
        plt.imshow(samples[i][0], cmap="gray")
    plt.show()
    import sys
    sys.exit()



#! Create Model--------------------------------------------------------------------------------------------------------
class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Relu = nn.ReLU()
        self.Linear2 = nn.Linear(hidden_size, num_classes)

    
    def forward(self, x):
        out = self.Linear1(x)
        out = self.Relu(out)
        out = self.Linear2(out)
        return out
    

model = MyNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()   # Kendisi zaten Softmax uygulayacak
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



#! Training------------------------------------------------------------------------------------------------------------
TOTAL_STEPS = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        ## Preprocess
        # Resize 100 x 1 x 28 x 28 => 100 x 784
        images = images.reshape(-1, 28*28).to(DEVICE)   # Yeniden boyutlandır ve GPU ile çalış
        labels = labels.to(DEVICE)                      # GPU ile çalış

        ## Forward
        predicted = model(images)
        loss = criterion(predicted, labels)             # Modelden çıkan "predicted" tensörü, torch'un graph tracking özelliği sayesinde, loss fonksiyonundan da geçtikten sonra hala takip halindedir. Yani modelin inputundan outputuna hatta devam eden işlemler boyunca izleme gerçekleşir.

        ## Backward
        optimizer.zero_grad()                           # Önceden hesaplanan gradyan değerlerinin toplanmaması için sıfırlama yapılır. Sıfırlanmazsa her "loss.backward()" ardından aynı ağırlıklar için eski ve yeni gradyan değerleri [weight].grad propertysi altında toplanır.
        loss.backward()                                 # Loss'un tüm ağırlıklara göre türevi alınır. [dLoss/dWeights] Gradyanların(graph tracking boyunca izlenen işlemlerin türev formülleri) değeri elde edilir.
        optimizer.step()                                # Her bir ağırlık için ölçülen türev sonuçlarına göre ağırlıklar güncellenir. Burada "Adam" optimizasyon metodu ile güncelleniyor.


        if(i+1) % 100 == 0:
            print(f"Epoch: {epoch+1} / {EPOCHS}, Step: {i+1}/{TOTAL_STEPS}, Loss: {loss.item():.4f}")


#! Test/Evaluate-------------------------------------------------------------------------------------------------------
import numpy as np

cm = np.zeros((10, 10))
with torch.no_grad():
    correct_predictions = 0
    number_samples = 0
    for images, labels in test_loader:
        ## Preprocess
        # Resize 100 x 1 x 28 x 28 => 100 x 784
        images = images.reshape(-1, 28*28).to(DEVICE)   # Yeniden boyutlandır ve GPU ile çalış
        labels = labels.to(DEVICE)                      # GPU ile çalış

        ## Evaluate
        predicted = model(images)                       # Evaluate
        _, predictions = torch.max(predicted, dim=1)    # Sonuçta 0-9 arası bir vectorde hangi çıkış en büyükse ona göre label sağlayacak. [value, index] = torch.max()
        number_samples += labels.shape[0]               # Batchteki örnek sayısı: 100
        correct_predictions += (predictions == labels).sum().item()
        
        ## Confussion Matrix
        cm += confusion_matrix(labels.to("cpu").numpy(), predictions.to("cpu").numpy(), labels=np.arange(10))

    ## Accuracy
    acc = 100.0 * (correct_predictions / number_samples)
    print(f"Accuracy: {acc}")


#! Visualize Confusion Matrix
figure = ConfusionMatrixDisplay(cm, display_labels=[str(x) for x in range(10)])
fig, ax = plt.subplots(figsize=(10, 10))
figure.plot(ax=ax)
plt.show()


# Extract Data From Confusion Matrix
results = CalculateConfusionMatrix(cm, 10, transpoze=False)