from datasets.dataset import CelebAGenderDataset
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CelebAGenderDataset(root_dir='data/celeba', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)