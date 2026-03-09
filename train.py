import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import SimpleCNN, ResNet18, ResNet34
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import AudioDataset
import argparse

def accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)

def train_single_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs, labels) * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_acc += accuracy(outputs, labels) * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', choices=['simplecnn', 'resnet18', 'resnet34'], help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='audio_ldm', choices=['audio_ldm', 'music_gen', 'music_ldm', 'mus_tango', 'stable_audio_open'], help='Dataset to use for training')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    train_dataset = AudioDataset([r'C:/Users/Apar2/OneDrive/Desktop/College/Spring 2026/capstone/frequency-print/data/FakeMusicCaps/audioldm2/*.wav'])  # Replace with actual file paths
    train_test_split = int(0.8 * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_test_split, len(train_dataset) - train_test_split])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = ResNet18(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25)