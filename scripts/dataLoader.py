from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.dataset import CelebAGenderDataset
import matplotlib.pyplot as plt
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CelebAGenderDataset(root_dir='data/celeba', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(loader))
print(images.shape, labels.shape)

fig, axes = plt.subplots(1, 4, figsize=(10, 4))
for i in range(4):
    img = images[i] 
    img = img.permute(1, 2, 0)   # CHW → HWC
    axes[i].imshow(img)
    axes[i].set_title("Male" if labels[i] == 1 else "Female")
    axes[i].axis("off")
plt.show()