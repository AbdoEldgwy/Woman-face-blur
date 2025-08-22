import torch.nn as nn
import torch.nn.functional as F
import torch

class GenderClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        # assuming input size 224*224
        self.fc1 = nn.Linear(64 * 112 * 112, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GenderClassifierModel:
    def __init__(self,model_path="outputs/gender_classifier.pth",device='cpu'):
        self.model = GenderClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

    def predict(self, img):
        with torch.no_grad():
            output = self.model(img.unsqueeze(0).to(self.device))
            _, predicted = torch.max(output, 1)
            return "female" if predicted.item() == 0 else "male" # 0 for female, 1 for male