import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import json
import torch.onnx

# Configuration
DATASET_PATH = 'datasets/Combined_Hieroglyphs/train'
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 20  # Increased for augmented training
MODEL_SAVE_PATH = 'app/public/hieroglyph_model.onnx'
LABELS_SAVE_PATH = 'app/src/model_labels.json'

# Define a simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 12 * 12, 512), # 100x100 -> 50 -> 25 -> 12.5 (12)
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model():
    print(f"Using device: {'cpu'}") # Force CPU for simplicity/compatibility on this env if unsure of MPS support
    device = torch.device('cpu') 

    # Check data
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    # Data Transforms - With MILD augmentation to handle real-world degraded images
    # Key: augmentation should be enough to generalize, but not destroy the signal
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(10),  # Reduced from 15
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Reduced
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1),  # Reduced
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),  # Only 30% of time
        transforms.ToTensor(),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    # Load Dataset - create two separate datasets with different transforms
    # First get class names from a base dataset
    base_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_transforms)
    class_names = base_dataset.classes
    print(f"Found {len(class_names)} classes.")
    
    # Save labels
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Saved labels to {LABELS_SAVE_PATH}")

    # Split indices first, then create datasets with appropriate transforms
    total_size = len(base_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Get random indices
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create augmented training dataset
    train_full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(base_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Acc: {epoch_acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%")

    # Export to ONNX
    print("Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, 100, 100, device=device)
    
    # Create public dir if not exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    torch.onnx.export(model, 
                      dummy_input, 
                      MODEL_SAVE_PATH, 
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
