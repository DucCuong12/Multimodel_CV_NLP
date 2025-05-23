import timm 
from transformers import BertModel
import torch.nn as nn 
from tqdm import tqdm
from process_data import MyData
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

train_dataset = MyData('kaggle/working/train_filtered/train_titles.csv', 'kaggle/working/train_filtered/train')
test_dataset = MyData('kaggle/working/test_filtered/test_titles.csv', 'kaggle/working/test_filtered/test')

# train_size = int(0.8 * len(df))
# test_size = len(df) - train_size
# train_dataset, test_dataset = random_split(df, [train_size, test_size])

train_data = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
test_data = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

print('hi')
class ImageEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224",return_all_tokens=False):
        super(ImageEncoder, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        
        # Optional: Remove classification head
        self.vit.head = nn.Identity()
        
        # Choose what to return
        self.return_all_tokens = return_all_tokens

    def forward(self, x):
        # x: [B, 3, 224, 224]
        if self.return_all_tokens:
            # Get all 197 tokens: [CLS] + 196 patch tokens
            tokens = self.vit.forward_features(x)  # Shape: [B, 197, 768]
            return tokens
        else:
            # Return only CLS token: [B, 768]
            cls_token = self.vit(x)
            return cls_token



class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
    def forward(self, input_ids, attention_mask):
        # tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.bert.device)
        output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        return output.last_hidden_state[:, 0, :]

class Single(nn.Module):
    def __init__(self):
        super(Single, self).__init__()
        self.img_encoder = ImageEncoder()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,30)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    def forward(self, img):
        img = self.img_encoder(img)
        layer1 = self.relu(self.fc1(img))
        layer2 = self.dropout(self.relu(self.fc2(layer1)))
        layer3 = self.fc3(layer2)
        return layer3


def test(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            img = batch['image'].to(device)
            # print(img.size())
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            output = model(img)
            predictions = output.argmax(dim=1)  # Get predicted labels
            correct += (predictions == label).sum().item()
    return correct/len(test_dataset)

def train1(model, dataloader, epochs = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    # scaler = GradScaler()



    for epoch in range(epochs):
        model.train()
        total_loss = 0 
        correct = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = batch['image'].to(device)
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(img)
                # print(output.size())
                if torch.isnan(output).any():
                    print("NaN in final output")
                if torch.isinf(output).any():
                    print("Inf in final output")
                if (output.abs() > 1e4).any():
                    print("⚠️ Warning: output too large", output.max(), output.min())
                loss = criterion(output, label) 
            
            prediction = output.argmax(dim = 1)
            correct+= (prediction == label).sum().item()
            loss.backward()

            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)  # cần thiết để clip_grad_norm đúng
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            total_loss += loss.item()
        result_test = test(model, test_data)
        # torch.save(model.state_dict(), f'epoch_{epoch}.pth')
        # print(f"Epoch {epoch+1} / {epochs}, Loss:{total_loss/ len(dataloader)}, Accuracy: {correct/train_size}, Test_accuracy: {result_test}")
        log_message = f"Epoch {epoch+1} / {epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {correct / len(train_dataset):.4f}, Test_accuracy: {result_test:.4f}, Test_size: {len(test_dataset)}"
        print(log_message)

# Save to log file
        with open("single_Img.txt", "a") as f:
            f.write(log_message + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Single()
model = torch.nn.DataParallel(model)
model.to(device)
print(model)
train1(model, train_data)
