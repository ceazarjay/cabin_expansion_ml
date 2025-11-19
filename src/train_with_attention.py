"""
Enhanced CNN with Attention Mechanism for Remote Sensing
Based on Adegun et al. (2023) - attention-based feature extraction
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import joblib
from pathlib import Path

from config import MODELS_DIR, DATA_DIR, METRICS_DIR

class AttentionModule(nn.Module):
    """Spatial attention mechanism for feature enhancement"""
    
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention_map = self.conv(x)
        attention_weights = self.sigmoid(attention_map)
        return x * attention_weights

class EnhancedFeatureExtractor(nn.Module):
    """CNN with attention for feature extraction from satellite imagery"""
    
    def __init__(self, num_features=128):
        super(EnhancedFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attention1 = AttentionModule(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention2 = AttentionModule(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attention3 = AttentionModule(128)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc(x)))
        
        return x

class AttentionEnhancedClassifier(nn.Module):
    """Complete model: Feature extractor + Classifier"""
    
    def __init__(self, num_classes=5):
        super(AttentionEnhancedClassifier, self).__init__()
        self.feature_extractor = EnhancedFeatureExtractor(num_features=128)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

class RemoteSensingDataset(Dataset):
    """Dataset for loading preprocessed satellite imagery"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_attention_model():
    """Train CNN with attention mechanism"""
    
    print("Training Attention-Enhanced CNN Model")
    print("Based on: Adegun et al. (2023) - Deep learning for remote sensing")
    
    X_train = np.load(DATA_DIR / 'processed' / 'X_train.npy')
    y_train = np.load(DATA_DIR / 'processed' / 'y_train.npy')
    X_val = np.load(DATA_DIR / 'processed' / 'X_val.npy')
    y_val = np.load(DATA_DIR / 'processed' / 'y_val.npy')
    X_test = np.load(DATA_DIR / 'processed' / 'X_test.npy')
    y_test = np.load(DATA_DIR / 'processed' / 'y_test.npy')
    
    if len(X_train.shape) == 2:
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        img_size = int(np.sqrt(n_features / 9))
        X_train = X_train.reshape(n_samples, 9, img_size, img_size)
        X_val = X_val.reshape(-1, 9, img_size, img_size)
        X_test = X_test.reshape(-1, 9, img_size, img_size)
    
    train_dataset = RemoteSensingDataset(X_train, y_train)
    val_dataset = RemoteSensingDataset(X_val, y_val)
    test_dataset = RemoteSensingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AttentionEnhancedClassifier(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    num_epochs = 50
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / 'attention_cnn_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(MODELS_DIR / 'attention_cnn_best.pth'))
    model.eval()
    
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(batch_y.numpy())
    
    test_acc = accuracy_score(test_true, test_preds)
    test_kappa = cohen_kappa_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average='weighted')
    
    print(f"\nFinal Test Results (Attention-Enhanced CNN):")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Kappa: {test_kappa:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    results = {
        'model': 'Attention-Enhanced CNN',
        'test_accuracy': test_acc,
        'test_kappa': test_kappa,
        'test_f1': test_f1,
        'reference': 'Adegun et al. (2023)'
    }
    
    joblib.dump(results, MODELS_DIR / 'attention_cnn_results.pkl')
    
    return model, results

if __name__ == "__main__":
    model, results = train_attention_model()
