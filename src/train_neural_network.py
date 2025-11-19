"""
Train Neural Network using PyTorch with GPU acceleration
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
import seaborn as sns

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    LAND_COVER_CLASSES, NN_PARAMS, RANDOM_STATE
)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

class LandCoverNN(nn.Module):
    """Neural Network for land cover classification"""
    
    def __init__(self, input_size, hidden_layers, n_classes, dropout_rate=0.3):
        super(LandCoverNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, n_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class NeuralNetworkTrainer:
    """Train and evaluate neural network"""
    
    def __init__(self):
        """Initialize and load data"""
        print("="*60)
        print("NEURAL NETWORK TRAINING (PyTorch + GPU)")
        print("="*60)
        
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load data (use scaled data for neural networks)
        print("\nLoading preprocessed data...")
        X_train = np.load(PROCESSED_DATA_DIR / 'X_train_scaled.npy')
        X_val = np.load(PROCESSED_DATA_DIR / 'X_val_scaled.npy')
        X_test = np.load(PROCESSED_DATA_DIR / 'X_test_scaled.npy')
        
        y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy')
        y_val = np.load(PROCESSED_DATA_DIR / 'y_val.npy')
        y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        
        self.y_train = torch.LongTensor(y_train).to(self.device)
        self.y_val = torch.LongTensor(y_val).to(self.device)
        self.y_test = torch.LongTensor(y_test).to(self.device)
        
        print(f"Data loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
        print(f"Input features: {X_train.shape[1]}")
        print(f"Number of classes: {len(LAND_COVER_CLASSES)}")
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=NN_PARAMS['batch_size'], 
            shuffle=True
        )
        
        val_dataset = TensorDataset(self.X_val, self.y_val)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=NN_PARAMS['batch_size'],
            shuffle=False
        )
    
    def train(self):
        """Train the neural network"""
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        
        # Initialize model
        input_size = self.X_train.shape[1]
        n_classes = len(LAND_COVER_CLASSES)
        
        self.model = LandCoverNN(
            input_size=input_size,
            hidden_layers=NN_PARAMS['hidden_layers'],
            n_classes=n_classes,
            dropout_rate=NN_PARAMS['dropout_rate']
        ).to(self.device)
        
        print(f"\nModel architecture:")
        print(self.model)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=NN_PARAMS['learning_rate'])
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        
        start_time = time.time()
        
        # Training loop
        print(f"\nTraining for up to {NN_PARAMS['epochs']} epochs...")
        print(f"Early stopping patience: {NN_PARAMS['patience']}")
        
        for epoch in range(NN_PARAMS['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / len(self.train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss = val_loss / len(self.val_loader)
            val_acc = val_correct / val_total
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NN_PARAMS['epochs']}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), MODELS_DIR / 'neural_network_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= NN_PARAMS['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(MODELS_DIR / 'neural_network_best.pth'))
        
        # Plot training history
        self.plot_training_history(history)
        
        return training_time, history
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*60)
        print("EVALUATION ON TEST SET")
        print("="*60)
        
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(self.X_test)
            _, predicted = torch.max(outputs.data, 1)
        
        # Convert to numpy for sklearn metrics
        y_true = self.y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Kappa: {kappa:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        print("\nPer-class metrics:")
        report = classification_report(
            y_true, y_pred,
            target_names=[LAND_COVER_CLASSES[i] for i in sorted(LAND_COVER_CLASSES.keys())],
            digits=4
        )
        print(report)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'kappa': float(kappa),
            'f1_weighted': float(f1),
            'classification_report': report
        }
    
    def plot_training_history(self, history):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'training_history_nn.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {FIGURES_DIR / 'training_history_nn.png'}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[LAND_COVER_CLASSES[i] for i in sorted(LAND_COVER_CLASSES.keys())],
            yticklabels=[LAND_COVER_CLASSES[i] for i in sorted(LAND_COVER_CLASSES.keys())],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Neural Network - Test Set Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'confusion_matrix_nn.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {FIGURES_DIR / 'confusion_matrix_nn.png'}")

def main():
    """Main training function"""
    trainer = NeuralNetworkTrainer()
    
    # Train model
    training_time, history = trainer.train()
    
    # Evaluate model
    test_metrics = trainer.evaluate()
    
    # Save results
    results = {
        'model': 'Neural Network (PyTorch)',
        'training_time': training_time,
        'test_metrics': test_metrics,
        'training_history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    with open(METRICS_DIR / 'neural_network_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("="*60)
    print(f"Model saved to: {MODELS_DIR / 'neural_network_best.pth'}")
    print(f"Results saved to: {METRICS_DIR / 'neural_network_results.json'}")

if __name__ == "__main__":
    main()
