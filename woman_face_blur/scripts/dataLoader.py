from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.dataset import CelebAGenderDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CelebAGenderDataset(root_dir='data/celeba', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check sample
images, labels = next(iter(loader))
print(images.shape, labels.shape)
