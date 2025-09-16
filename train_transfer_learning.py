
"""
train_transfer_learning.py

Transfer Learning pipeline for Poultry Disease Classification (PyTorch).
Usage (example):
    python train_transfer_learning.py --data_dir /path/to/dataset --output_dir ./output --epochs 10 --batch_size 32 --lr 1e-4
Dataset format: expects ImageFolder structure:
    data_dir/
        train/
            class1/
            class2/
            ...
        val/
            class1/
            class2/
            ...
"""

import argparse
import os
from pathlib import Path
import time
import copy
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset (ImageFolder format)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Where to save models and results')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18','resnet50','vgg16','densenet121'], help='Pretrained model to use')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights and train classifier only')
    return parser.parse_args()

def get_transforms(input_size=224):
    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return train_transforms, val_transforms

def create_model(model_name, num_classes, pretrained=True, freeze_backbone=False):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError('Unsupported model: ' + model_name)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not any([p in name for p in ['fc','classifier']]):
                param.requires_grad = False
    return model

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, output_dir):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-'*20)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total

            history[f'{phase}_loss' if phase in ['train','val'] else phase + '_loss'].append(epoch_loss)
            history[f'{phase}_acc' if phase in ['train','val'] else phase + '_acc'].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s. Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    # save history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history

def evaluate_model(model, dataloader, device, class_names, output_dir):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

    report = classification_report(trues, preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(trues, preds)
    # Save
    import numpy as np
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        import json
        json.dump(report, f, indent=2)
    print("Classification report:")
    print(classification_report(trues, preds, target_names=class_names))
    print("Confusion matrix:")
    print(cm)

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transforms, val_transforms = get_transforms(input_size=224)

    image_datasets = {
        'train': datasets.ImageFolder(root=str(data_dir / 'train'), transform=train_transforms),
        'val': datasets.ImageFolder(root=str(data_dir / 'val'), transform=val_transforms)
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=4)
    }
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f'Found classes: {class_names}')

    device = torch.device(args.device)
    model = create_model(args.model, num_classes=num_classes, pretrained=True, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # Only train parameters that require grad
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params_to_update, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model, history = train_model(model, dataloaders, criterion, optimizer, device, args.epochs, str(output_dir))
    evaluate_model(model, dataloaders['val'], device, class_names, str(output_dir))

    # Save class mapping
    with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, indent=2)

if __name__ == '__main__':
    main()
