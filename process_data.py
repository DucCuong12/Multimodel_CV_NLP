import os 
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import BertTokenizer
import html

class MyData(Dataset):
    def __init__(self, csv_file, img_file):
        self.csv_file = csv_file
        self.img_file = img_file 
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.dataset = pd.read_csv(self.csv_file)
        self.label_map = {label: idx for idx, label in enumerate(self.dataset['label'].unique())}
        # self.dataset = pd.read_csv(self.csv_file, encoding="latin1")
        self.max_length = 512
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        label_text = item['label']
    #######  IMAGE ######
        img_path = os.path.join(self.img_file, label_text, str(item['image']))
        if not os.path.exists(img_path):
            print(f"File not found {img_path}")
            label_dir = os.path.join(self.img_file,label_text)
            images = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.isdir(label_dir) else []
            img_path = os.path.join(label_dir, images[0]) if images else None
        # print(img_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
    ####### TEXT ########
        
        # tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.bert.device)

        # text = str(item['Crock-Pot Ladies  Crock-Pot Apple Pie Moonshine'])
        text = html.unescape(str(item['text']))
        tokens = self.tokenizer(text, padding='max_length', truncation=True,  max_length=self.max_length, return_tensors="pt")
       
        # output = self.bert(input_ids = tokens["input_ids"], attention_mask = tokens["attention_mask"])

        label = self.label_map[label_text]
        return {"image": image, "input_ids": tokens["input_ids"].squeeze(0), "attention_mask":  tokens["attention_mask"].squeeze(0),  "label": label}