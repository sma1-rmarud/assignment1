import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import Model
from tqdm import tqdm
import logging
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("weights", exist_ok=True)

logging.basicConfig(
    filename="logs/train_mnist.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch_idx, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()
        _, predicted = outputs.max(1)
        correct = correct + (predicted == labels).sum().item()
        total = total + labels.size(0)

        progress_bar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}", acc=f"{100 * correct / total:.2f}%")
        progress_bar.update(1)

    progress_bar.close()

    log_str = (f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    logging.info(log_str)

torch.save(model.state_dict(), "weights/model_for_mnist.pth")
logging.info("Model saved: weights/model_for_mnist.pth")