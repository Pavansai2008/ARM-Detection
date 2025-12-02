import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os

if __name__ == '__main__':
    # Define device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Define transforms with enhanced augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),  # Increased rotation range
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 2: Load datasets from balanced_dataset/
    try:
        train_dataset = datasets.ImageFolder('balanced_dataset/train', transform=train_transform)
        val_dataset = datasets.ImageFolder('balanced_dataset/val', transform=val_test_transform)
        test_dataset = datasets.ImageFolder('balanced_dataset/test', transform=val_test_transform)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit()

    # Step 3: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Step 4: Load pre-trained ResNet18 and modify it with dropout
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    # Add dropout before the final layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # 50% dropout to prevent overfitting
        nn.Linear(num_ftrs, 2)  # 2 classes: ultrasound, non-ultrasound
    )
    model = model.to(device)

    # Step 5: Define loss function and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Reduced lr, added weight decay

    # Step 6: Training loop with early stopping
    num_epochs = 20  # Increased to allow early stopping to work
    best_val_loss = float('inf')
    patience = 3  # Stop if no improvement for 3 epochs
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            try:
                images, labels = images.to(device), labels.to(device)
                print(f"Image tensor shape: {images.shape}")
                print(f"Image tensor dtype: {images.dtype}")
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            except Exception as e:
                print(f"Error during training batch: {e}")
                continue
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                except Exception as e:
                    print(f"Error during validation batch: {e}")
                    continue
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model and implement early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_ultrasound_classifier.pth')
            print("Saved best model with Val Loss: {:.4f}".format(best_val_loss))
            patience_counter = 0  # Reset counter
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Step 7: Evaluate on test set using the best model
    try:
        model.load_state_dict(torch.load('best_ultrasound_classifier.pth', map_location=device))
    except Exception as e:
        print(f"Error loading best model: {e}")
        exit()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            try:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception as e:
                print(f"Error during test batch: {e}")
                continue

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Step 8: Save the final model
    torch.save(model.state_dict(), 'ultrasound_classifier.pth')
    print("Final model saved as 'ultrasound_classifier.pth'")