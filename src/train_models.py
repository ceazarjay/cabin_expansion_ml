"""
Train and evaluate machine learning models for land cover classification
"""
import numpy as np
import pandas as pd
import joblib
import json
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    cohen_kappa_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR,
    LAND_COVER_CLASSES, RF_PARAMS, SVM_PARAMS, RANDOM_STATE
)

class ModelTrainer:
    """Train and evaluate classical ML models"""
    
    def __init__(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        # Load unscaled data for Random Forest
        self.X_train = np.load(PROCESSED_DATA_DIR / 'X_train.npy')
        self.X_val = np.load(PROCESSED_DATA_DIR / 'X_val.npy')
        self.X_test = np.load(PROCESSED_DATA_DIR / 'X_test.npy')
        
        # Load scaled data for SVM
        self.X_train_scaled = np.load(PROCESSED_DATA_DIR / 'X_train_scaled.npy')
        self.X_val_scaled = np.load(PROCESSED_DATA_DIR / 'X_val_scaled.npy')
        self.X_test_scaled = np.load(PROCESSED_DATA_DIR / 'X_test_scaled.npy')
        
        # Load labels
        self.y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy')
        self.y_val = np.load(PROCESSED_DATA_DIR / 'y_val.npy')
        self.y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')
        
        print(f"Data loaded: {len(self.X_train)} train, {len(self.X_val)} val, {len(self.X_test)} test")
    
    def train_random_forest(self):
        """Train Random Forest classifier"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        start_time = time.time()
        
        # Train model
        print("Training...")
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set
        print("\nValidation Results:")
        val_pred = rf.predict(self.X_val)
        val_metrics = self.calculate_metrics(self.y_val, val_pred, "Random Forest - Validation")
        
        # Evaluate on test set
        print("\nTest Results:")
        test_pred = rf.predict(self.X_test)
        test_metrics = self.calculate_metrics(self.y_test, test_pred, "Random Forest - Test")
        
        # Feature importance
        feature_names = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI']
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df.to_string(index=False))
        
        # Save model
        joblib.dump(rf, MODELS_DIR / 'random_forest.pkl')
        print(f"\nModel saved to: {MODELS_DIR / 'random_forest.pkl'}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            self.y_test, test_pred, 
            "Random Forest - Test Set",
            FIGURES_DIR / 'confusion_matrix_rf.png'
        )
        
        # Plot feature importance
        self.plot_feature_importance(
            importance_df,
            FIGURES_DIR / 'feature_importance_rf.png'
        )
        
        return {
            'model': 'Random Forest',
            'training_time': training_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': importance_df.to_dict()
        }
    
    def train_svm(self):
        """Train Support Vector Machine classifier"""
        print("\n" + "="*60)
        print("TRAINING SUPPORT VECTOR MACHINE")
        print("="*60)
        
        start_time = time.time()
        
        # Train model (using scaled data)
        print("Training...")
        svm = SVC(**SVM_PARAMS, probability=True)
        svm.fit(self.X_train_scaled, self.y_train)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set
        print("\nValidation Results:")
        val_pred = svm.predict(self.X_val_scaled)
        val_metrics = self.calculate_metrics(self.y_val, val_pred, "SVM - Validation")
        
        # Evaluate on test set
        print("\nTest Results:")
        test_pred = svm.predict(self.X_test_scaled)
        test_metrics = self.calculate_metrics(self.y_test, test_pred, "SVM - Test")
        
        # Save model
        joblib.dump(svm, MODELS_DIR / 'svm.pkl')
        print(f"\nModel saved to: {MODELS_DIR / 'svm.pkl'}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            self.y_test, test_pred,
            "SVM - Test Set",
            FIGURES_DIR / 'confusion_matrix_svm.png'
        )
        
        return {
            'model': 'SVM',
            'training_time': training_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    
    def calculate_metrics(self, y_true, y_pred, title):
        """Calculate and print classification metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\n{title}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Kappa: {kappa:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        
        print("\nPer-class metrics:")
        report = classification_report(
            y_true, y_pred, 
            target_names=[LAND_COVER_CLASSES[i] for i in sorted(LAND_COVER_CLASSES.keys())],
            digits=4
        )
        print(report)
        
        return {
            'accuracy': float(accuracy),
            'kappa': float(kappa),
            'f1_weighted': float(f1),
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, title, filepath):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[LAND_COVER_CLASSES[i] for i in sorted(LAND_COVER_CLASSES.keys())],
            yticklabels=[LAND_COVER_CLASSES[i] for i in sorted(LAND_COVER_CLASSES.keys())],
            cbar_kws={'label': 'Count'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {filepath}")
    
    def plot_feature_importance(self, importance_df, filepath):
        """Plot and save feature importance"""
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to: {filepath}")
    
    def compare_models(self, results):
        """Create comparison table of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Model': result['model'],
                'Training Time (s)': f"{result['training_time']:.2f}",
                'Val Accuracy': f"{result['val_metrics']['accuracy']:.4f}",
                'Val Kappa': f"{result['val_metrics']['kappa']:.4f}",
                'Val F1': f"{result['val_metrics']['f1_weighted']:.4f}",
                'Test Accuracy': f"{result['test_metrics']['accuracy']:.4f}",
                'Test Kappa': f"{result['test_metrics']['kappa']:.4f}",
                'Test F1': f"{result['test_metrics']['f1_weighted']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv(METRICS_DIR / 'model_comparison.csv', index=False)
        print(f"\nComparison table saved to: {METRICS_DIR / 'model_comparison.csv'}")
        
        # Save detailed results as JSON
        with open(METRICS_DIR / 'all_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {METRICS_DIR / 'all_results.json'}")

def main():
    """Main training function"""
    print("="*60)
    print("LAND COVER CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    trainer = ModelTrainer()
    
    results = []
    
    # Train Random Forest
    rf_results = trainer.train_random_forest()
    results.append(rf_results)
    
    # Train SVM
    svm_results = trainer.train_svm()
    results.append(svm_results)
    
    # Compare models
    trainer.compare_models(results)
    
    print("\n" + "="*60)
    print("="*60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")

if __name__ == "__main__":
    main()
