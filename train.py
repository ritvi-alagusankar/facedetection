import os
import pickle
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm

# Initialize FaceNet model (for embeddings) - CPU only
facenet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=40, keep_all=False, thresholds=[0.6, 0.7, 0.7], min_face_size=40)

# Force CPU usage
device = torch.device("cpu")

# Enhanced Data Augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(160, scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomAutocontrast(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

# Validation transformations
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Custom collate function to handle None values
def custom_collate_fn(batch):
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))
    
    # If all elements are None, return empty lists
    if len(batch) == 0:
        return torch.Tensor([]), []
    
    # Separate images and labels
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, labels

# Dataset Class
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, preload=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preload = preload
        self.preloaded_faces = {}
        
        # Pre-process images to find valid faces
        self.valid_indices = []
        print("Pre-processing images to detect faces...")
        
        for idx, img_path in tqdm(enumerate(self.image_paths), total=len(self.image_paths)):
            try:
                img = Image.open(img_path).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    self.valid_indices.append(idx)
                    if preload:
                        face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        self.preloaded_faces[idx] = face_np
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
        print(f"Found {len(self.valid_indices)} valid faces out of {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        true_idx = self.valid_indices[idx]
        img_path = self.image_paths[true_idx]
        label = self.labels[true_idx]
        
        try:
            if self.preload and true_idx in self.preloaded_faces:
                face_np = self.preloaded_faces[true_idx]
            else:
                img = Image.open(img_path).convert('RGB')
                face = mtcnn(img)
                
                # This should not happen since we prefiltered, but just in case
                if face is None:
                    print(f"Warning: Face detection failed for {img_path} during training")
                    # Return a placeholder tensor and the label
                    placeholder = torch.zeros(3, 160, 160)
                    return placeholder, label
                
                face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            if self.transform:
                face_tensor = self.transform(face_np)
            else:
                face_tensor = transforms.ToTensor()(face_np)
            
            return face_tensor, label
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Return a placeholder tensor and the label
            placeholder = torch.zeros(3, 160, 160)
            return placeholder, label

# Load Dataset
def load_dataset(dataset_path, val_size=0.15, test_size=0.15):
    image_paths, labels = [], []
    class_names = sorted(os.listdir(dataset_path))
    
    print(f"Found classes: {class_names}")
    
    # Count examples per class for weighted loss calculation
    class_counts = {class_name: 0 for class_name in class_names}
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        image_count = 0
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, file))
                labels.append(class_name)
                image_count += 1
                class_counts[class_name] += 1
        
        print(f"Class '{class_name}': {image_count} images")
    
    if len(image_paths) == 0:
        raise ValueError("No images found in the dataset directory. Please check the path and image formats.")
    
    # First split: separate training+validation from test set
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels)
    
    # Second split: separate training from validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, 
        test_size=val_size/(1-test_size),  # Adjust validation size relative to remaining data
        random_state=42, 
        stratify=train_val_labels)
    
    print(f"Total images: {len(image_paths)}")
    print(f"Training: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Testing: {len(test_images)}")
    
    # Calculate class weights for weighted loss
    class_weights = {}
    total_samples = sum(class_counts.values())
    for class_name, count in class_counts.items():
        class_weights[class_name] = total_samples / (len(class_counts) * count)
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels, class_names, class_weights

# Enhanced FaceClassifier Model with dropout and batch normalization
class FaceClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout_rate=0.5):
        super(FaceClassifier, self).__init__()
        
        # First dense block
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second dense block
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # First block
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer (no softmax - will be part of loss function)
        x = self.fc3(x)
        
        return x

# Create a simple text-based confusion matrix
def create_confusion_matrix(true_labels, pred_labels, class_names, save_path="confusion_matrix.txt"):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, pred_labels)
    with open(save_path, 'w') as f:
        # Write header
        f.write("Confusion Matrix:\n\n")
        f.write("True\\Pred\t" + "\t".join(class_names) + "\n")
        
        # Write rows
        for i, class_name in enumerate(class_names):
            row = [str(x) for x in cm[i]]
            f.write(f"{class_name}\t" + "\t".join(row) + "\n")
    
    print(f"Confusion matrix saved to {save_path}")

# Evaluate model on validation set
def evaluate_model(model, val_loader, criterion, device, label_map):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            if images.shape[0] == 0:
                continue
                
            images = images.to(device)
            label_indices = torch.tensor([label_map[label] for label in labels]).to(device)
            
            embeddings = facenet(images)
            outputs = model(embeddings)
            loss = criterion(outputs, label_indices)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label_indices.size(0)
            correct += (predicted == label_indices).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy, all_preds, all_labels

# Save training history to text file
def save_training_history(train_losses, val_losses, train_accs, val_accs, filename="training_history.txt"):
    with open(filename, 'w') as f:
        f.write("Epoch\tTrain Loss\tVal Loss\tTrain Acc\tVal Acc\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1}\t{train_losses[i]:.4f}\t{val_losses[i]:.4f}\t{train_accs[i]:.4f}\t{val_accs[i]:.4f}\n")
    print(f"Training history saved to {filename}")

if __name__ == "__main__":
    # Set the correct dataset path
    dataset_path = "dataset"
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available directories: {os.listdir('.')}")
        exit(1)
    
    # Load dataset with validation split
    train_images, train_labels, val_images, val_labels, test_images, test_labels, class_names, class_weights = load_dataset(dataset_path)
    
    # Create Dataset Loaders
    train_dataset = FaceDataset(train_images, train_labels, transform=train_transform)
    val_dataset = FaceDataset(val_images, val_labels, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=2,
        collate_fn=custom_collate_fn
    )

    num_classes = len(class_names)
    label_map = {label: idx for idx, label in enumerate(class_names)}
    
    # Create weight tensor for weighted loss
    weight_tensor = torch.zeros(num_classes)
    for label, idx in label_map.items():
        weight_tensor[idx] = class_weights[label]

    # Initialize improved model
    model = FaceClassifier(embedding_dim=512, num_classes=num_classes).to(device)
    
    # Use weighted loss if there's class imbalance
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    
    # Better optimizer with weight decay to reduce overfitting
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Keep track of learning rate changes
    current_lr = optimizer.param_groups[0]['lr']

    # Training parameters
    num_epochs = 30
    best_val_accuracy = 0
    best_model_path = "best_face_model.pkl"
    
    # Lists to track metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training Loop with validation
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct, total = 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in progress_bar:
            # Skip empty batches
            if images.shape[0] == 0:
                continue
            
            images = images.to(device)
            label_indices = torch.tensor([label_map[label] for label in labels]).to(device)
            
            with torch.no_grad():
                embeddings = facenet(images)
            
            outputs = model(embeddings)
            loss = criterion(outputs, label_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label_indices.size(0)
            correct += (predicted == label_indices).sum().item()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.4f}"})
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        val_loss, val_acc, val_preds, val_true = evaluate_model(model, val_loader, criterion, device, label_map)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate based on validation loss
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Manually report if learning rate changed
        if new_lr != prev_lr:
            print(f"Learning rate changed from {prev_lr:.6f} to {new_lr:.6f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Current LR: {new_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            model_data = {
                'model': model.state_dict(),
                'class_names': class_names,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'epoch': epoch
            }
            with open(best_model_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
    
    # Save training history as text file instead of plots
    save_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Create confusion matrix for best model
    if os.path.exists(best_model_path):
        try:
            best_model_data = pickle.load(open(best_model_path, "rb"))
            model.load_state_dict(best_model_data['model'])
            print(f"Loaded best model from epoch {best_model_data['epoch']} with validation accuracy {best_model_data['val_acc']:.4f}")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Using the current model state for final evaluation")
    else:
        print("Best model file not found, using the current model state for final evaluation")
    
    _, _, val_preds, val_true = evaluate_model(model, val_loader, criterion, device, label_map)
    
    if len(val_preds) > 0 and len(val_true) > 0:
        create_confusion_matrix([class_names[i] for i in val_true], 
                              [class_names[i] for i in val_preds], 
                              class_names)
    
    # Save final model
    with open("face_model.pkl", "wb") as f:
        pickle.dump({'model': model.state_dict(), 'class_names': class_names}, f)
    
    print(f"Model trained and saved successfully. Best validation accuracy: {best_val_accuracy:.4f}")
