import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as DataLoader

from datasets.dataset import CelebAGenderDataset
from models.classifier import GenderClassifier

# Dataset
train_data = CelebAGenderDataset('data/celeba')
train_loader = DataLoader.DataLoader(train_data, batch_size=32, shuffle=True)

# Model
model = GenderClassifier(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training 
for epoch in range(10):
    for img, label in train_loader:
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item()}")

# save model
torch.save(model.state_dict(), 'outputs/gender_classifier.pth')