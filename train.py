import torch
import torch.optim as optim
import torch.nn as nn
from process_data import MyData 
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from Contrastive import Q_cons_fusion
from torch.optim import AdamW, Adam
import os
from MLP import MLP_fusion
from torch.cuda.amp import autocast, GradScaler
from attention_model import AttentionModule
from tqdm import tqdm
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

train_dataset = MyData('kaggle/working/train_filtered/train_titles.csv', 'kaggle/working/train_filtered/train')
test_dataset = MyData('kaggle/working/test_filtered/test_titles.csv', 'kaggle/working/test_filtered/test')

# train_size = int(0.8 * len(df))
# test_size = len(df) - train_size
# train_dataset, test_dataset = random_split(df, [train_size, test_size])

train_data = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
test_data = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

print('hi')

def train(model, dataloader, epochs = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([
    {
        'params': model.module.text_encoder.bert.parameters(),  # thêm .module
        'lr': 2e-5,
        'weight_decay': 0.01
    },
    {
        'params': model.module.q_mlp.parameters(),
        'lr': 5e-4,
        'weight_decay': 0.01
    },
    {
        'params': model.module.fusion.parameters(),
        'lr': 5e-4,
        'weight_decay': 0.01
    },
    {
        'params': model.module.cross_Q.parameters(),
        'lr': 1e-3,
        'weight_decay': 0.01
    }
])
    # scaler = GradScaler()



    for epoch in range(epochs):
        model.train()
        total_loss = 0 
        correct = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(img, input_ids, attention_mask)
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
        log_message = f"Epoch {epoch+1} / {epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {correct / len(train_dataset):.4f}, Test_accuracy: {result_test:.4f}"
        print(log_message)

# Save to log file
        with open("-constrastive.txt", "a") as f:
            f.write(log_message + "\n")
def test(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            img = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            output = model(img, input_ids, attention_mask)
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(img, input_ids, attention_mask)
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
        with open("attention.txt", "a") as f:
            f.write(log_message + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionModule()
model = torch.nn.DataParallel(model)
model.to(device)
print(model)
train1(model, train_data)

    
