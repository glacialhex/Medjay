"""
ResNet18-based Hieroglyph Classifier with Transfer Learning
Uses pre-trained ImageNet weights and class-weighted loss for imbalanced data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import os
import json
import numpy as np
from collections import Counter

# Configuration
DATASET_PATH = 'datasets/Combined_Hieroglyphs/train'
IMG_SIZE = (224, 224)  # ResNet expects 224x224
BATCH_SIZE = 32
EPOCHS = 30
MODEL_SAVE_PATH = 'app/public/hieroglyph_model.onnx'
LABELS_SAVE_PATH = 'app/src/model_labels.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_resnet_model(num_classes):
    """Create ResNet18 with pre-trained weights, replace final layer"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers (keep pre-trained features)
    for param in list(model.parameters())[:-20]:  # Freeze all but last few layers
        param.requires_grad = False
    
    # Replace final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def compute_class_weights(dataset):
    """Compute class weights inversely proportional to frequency"""
    targets = [dataset.targets[i] for i in range(len(dataset))]
    class_counts = Counter(targets)
    total = len(targets)
    num_classes = len(class_counts)
    
    # Inverse frequency with smoothing
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        # Use sqrt for less aggressive weighting
        weight = np.sqrt(total / (num_classes * count))
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.mean()
    
    print(f"Class weight range: {weights.min():.2f} to {weights.max():.2f}")
    return torch.FloatTensor(weights)

def train_model():
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return
    
    # Data Transforms - ResNet expects normalized 224x224 RGB
    # Use ImageNet normalization 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Custom transform to simulate stone-texture photos
    class StonifyTransform:
        """Simulates photos of hieroglyphs carved in stone"""
        def __init__(self, p=0.5):
            self.p = p
        
        def __call__(self, img):
            import random
            from PIL import ImageFilter, ImageEnhance
            
            if random.random() > self.p:
                return img
            
            # Random contrast reduction (0.3 to 0.7)
            contrast = random.uniform(0.3, 0.7)
            img = ImageEnhance.Contrast(img).enhance(contrast)
            
            # Random blur (radius 1-3)
            blur_radius = random.uniform(0.5, 2.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Reduce color saturation (stone is grayish)
            color = random.uniform(0.2, 0.5)
            img = ImageEnhance.Color(img).enhance(color)
            
            # Random brightness adjustment
            brightness = random.uniform(0.7, 1.3)
            img = ImageEnhance.Brightness(img).enhance(brightness)
            
            return img
    
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        StonifyTransform(p=0.5),  # 50% chance of stone-like degradation
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.1),  # Some glyphs may be symmetric
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load full dataset to get class info and compute weights
    full_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")
    
    # Save labels
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Saved labels to {LABELS_SAVE_PATH}")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(full_dataset).to(DEVICE)
    
    # Split indices
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets with different transforms
    train_full = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
    train_dataset = torch.utils.data.Subset(train_full, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                                shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                                              shuffle=False, num_workers=0)
    
    # Create model
    model = create_resnet_model(num_classes).to(DEVICE)
    print(f"Model: ResNet18 with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Weighted cross-entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer - different learning rates for pretrained vs new layers
    optimizer = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': 1e-3},  # New layers learn faster
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad], 'lr': 1e-4}
    ])
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
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
        per_class_correct = {}
        per_class_total = {}
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Track per-class accuracy
                for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                    if label not in per_class_total:
                        per_class_total[label] = 0
                        per_class_correct[label] = 0
                    per_class_total[label] += 1
                    if pred == label:
                        per_class_correct[label] += 1
        
        val_acc = 100 * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%")
        
        # Track worst performing classes
        if epoch == EPOCHS - 1 or val_acc > best_val_acc:
            worst_classes = []
            for class_idx in per_class_total:
                acc = 100 * per_class_correct[class_idx] / per_class_total[class_idx]
                if acc < 50:  # Flag classes under 50%
                    worst_classes.append((class_names[class_idx], acc, per_class_total[class_idx]))
            
            if worst_classes:
                worst_classes.sort(key=lambda x: x[1])
                print(f"  Worst classes (epoch {epoch+1}):")
                for name, acc, count in worst_classes[:5]:
                    print(f"    {name}: {acc:.0f}% ({count} samples)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model checkpoint
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for export
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Export to ONNX
    print("Exporting to ONNX...")
    model.eval()
    model.to('cpu')  # Export on CPU for compatibility
    dummy_input = torch.randn(1, 3, 224, 224)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_SAVE_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Cleanup
    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')

if __name__ == '__main__':
    train_model()
