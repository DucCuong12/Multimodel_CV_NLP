import os 
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
class MyData(Dataset):
    def __init__(self, csv_file, img_file):
        self.csv_file = csv_file
        self.img_file = img_file 
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.0, 0.0, 0.0], std = [1.0, 1.0, 1.0])
            ]
        )
        self.dataset = pd.read_csv(self.csv_file, delimiter=";", encoding="latin1")
        self.dataset['img_path'] = self.dataset['img_path'].str.replace(r'\\', '/', regex=True)
        # self.dataset = pd.read_csv(self.csv_file, encoding="latin1")

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        self.dataset.loc[idx, 'img_path'] = str(self.dataset.loc[idx, 'img_path']).replace('imgamazon/', '')
        item = self.dataset.iloc[idx]
    #######  IMAGE ######
        img_path = os.path.join(self.img_file, str(item['img_path']))
        if not os.path.exists(img_path):
            print(f"File not found {img_path}")
            img_path = "amazon_dataset/31bNhi6E3eL._AC_.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
    ####### TEXT ########

        text = str(item['description'])
        label = torch.tensor(int(item["label_id"])-1, dtype = torch.long)
        return {"image": image, "text": text, "label": label}
        
