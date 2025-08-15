import os 
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class CelebAGenderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        attr_path = os.path.join(root_dir, 'list_attr_celeba.csv')
        if not os.path.exists(attr_path):
            raise FileNotFoundError(f"Attribute file not found at {attr_path}")
        
        self.attr_df = pd.read_csv(attr_path)
        self.attr_df.columns = self.attr_df.columns.str.strip()

        self.attr_df = self.attr_df[['image_id', 'Male']]

        # Convert labels: 1 = male, -1 = female
        self.attr_df['Male'] = (self.attr_df['Male'] == 1).astype(int)

        self.records = list(zip(self.attr_df['image_id'], self.attr_df['Male']))


    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        fname, label = self.records[idx]
        img_path = os.path.join(self.root_dir, 'img_align_celeba', fname)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
