"""
Apply trained models to detect land cover changes between 2019 and 2024
Quantify cabin expansion in Trysil region
"""
import numpy as np
import rioxarray as rxr
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
    FIGURES_DIR, RESULTS_DIR, LAND_COVER_CLASSES
)

class ChangeDetector:
    """Detect land cover changes using trained models"""
    
    def __init__(self, model_name='random_forest'):
        """
        Initialize change detector
        
        Args:
            model_name (str): Name of model to use ('random_forest', 'svm', 'neural_network')
        """
        print("="*70)
        print(f"CHANGE DETECTION - {model_name.upper()}")
        print("="*70)
        
        self.model_name = model_name
        
        # Load the trained model
        print(f"\nLoading {model_name} model...")
        if model_name == 'random_forest':
            self.model = joblib.load(MODELS_DIR / 'random_forest.pkl')
            self.scaler = None  # RF doesn't need scaling
        elif model_name == 'svm':
            self.model = joblib.load(MODELS_DIR / 'svm.pkl')
            self.scaler = joblib.load(PROCESSED_DATA_DIR / 'scaler.pkl')
        else:
            raise ValueError(f"Model {model_name} not supported for change detection")
        
        
        # Color map for visualization
        self.colors = ['#2E86AB', '#2D5016', '#90D950', '#E63946', '#8B4513']
        self.cmap = ListedColormap(self.colors)
        
    def load_and_preprocess_image(self, filepath):
        """
        Load GeoTIFF and prepare features
        
        Args:
            filepath (Path): Path to GeoTIFF file
            
        Returns:
            features (np.ndarray): Feature array (n_pixels, n_features)
            shape (tuple): Original image shape (height, width)
            profile (dict): Raster metadata
        """
        print(f"\nLoading: {filepath.name}")
        
        # Load with rioxarray
        data = rxr.open_rasterio(filepath)
        
        # Get dimensions
        n_bands, height, width = data.shape
        print(f"  Dimensions: {height} x {width} pixels, {n_bands} bands")
        
        # Convert to numpy and reshape to (n_pixels, n_bands)
        image_array = data.values
        features = image_array.reshape(n_bands, -1).T
        
        # Get spatial reference info
        profile = {
            'transform': data.rio.transform(),
            'crs': data.rio.crs,
            'width': width,
            'height': height
        }
        
        # Handle NaN values (replace with 0)
        nan_mask = np.isnan(features).any(axis=1)
        features[nan_mask] = 0
        
        print(f"  Valid pixels: {(~nan_mask).sum()} / {len(features)}")
        
        return features, (height, width), profile, nan_mask
    
    def classify_image(self, features, shape, nan_mask):
        """
        Classify all pixels in an image
        
        Args:
            features (np.ndarray): Feature array (n_pixels, n_features)
            shape (tuple): Original image shape (height, width)
            nan_mask (np.ndarray): Boolean mask of NaN pixels
            
        Returns:
            classification (np.ndarray): Classified image (height, width)
        """
        print("  Classifying pixels...")
        
        # Apply scaling if needed (for SVM)
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # Predict all pixels
        predictions = self.model.predict(features_scaled)
        
        # Set NaN pixels to -1 (no data)
        predictions[nan_mask] = -1
        
        # Reshape to original image dimensions
        classification = predictions.reshape(shape)
        
        
        return classification
    
    def detect_changes(self, class_2019, class_2024, shape):
        """
        Detect changes between two classifications
        
        Args:
            class_2019 (np.ndarray): 2019 classification
            class_2024 (np.ndarray): 2024 classification
            shape (tuple): Image shape
            
        Returns:
            change_map (np.ndarray): Map of changes
            change_stats (dict): Statistics of changes
        """
        print("\nDetecting changes...")
        
        # Calculate change matrix
        change_stats = {}
        
        # Count pixels per class for each year
        for year, classification in [('2019', class_2019), ('2024', class_2024)]:
            counts = {}
            for class_id, class_name in LAND_COVER_CLASSES.items():
                count = np.sum(classification == class_id)
                counts[class_name] = int(count)
            change_stats[year] = counts
        
        # Calculate pixel area (10m x 10m = 100 m² = 0.0001 km²)
        pixel_area_km2 = 0.0001
        
        # Calculate area changes
        area_changes = {}
        for class_name in LAND_COVER_CLASSES.values():
            pixels_2019 = change_stats['2019'][class_name]
            pixels_2024 = change_stats['2024'][class_name]
            change_pixels = pixels_2024 - pixels_2019
            change_km2 = change_pixels * pixel_area_km2
            change_percent = (change_pixels / pixels_2019 * 100) if pixels_2019 > 0 else 0
            
            area_changes[class_name] = {
                'pixels_2019': pixels_2019,
                'pixels_2024': pixels_2024,
                'change_pixels': change_pixels,
                'area_2019_km2': pixels_2019 * pixel_area_km2,
                'area_2024_km2': pixels_2024 * pixel_area_km2,
                'change_km2': change_km2,
                'change_percent': change_percent
            }
        
        # Create change map (focus on Built_up expansion)
        change_map = np.zeros(shape, dtype=int)
        
        # 0: No change
        # 1: New built-up (any class → Built_up)
        # 2: Lost built-up (Built_up → any class)
        # 3: Other changes
        
        built_up_id = 3  # Built_up class ID
        
        change_map[(class_2019 != built_up_id) & (class_2024 == built_up_id)] = 1  # New built-up
        change_map[(class_2019 == built_up_id) & (class_2024 != built_up_id)] = 2  # Lost built-up
        change_map[(class_2019 != class_2024) & (change_map == 0)] = 3  # Other changes
        
        return change_map, area_changes
    
    def visualize_results(self, class_2019, class_2024, change_map, area_changes):
        """
        Create visualizations of results
        
        Args:
            class_2019 (np.ndarray): 2019 classification
            class_2024 (np.ndarray): 2024 classification  
            change_map (np.ndarray): Change detection map
            area_changes (dict): Change statistics
        """
        print("\nGenerating visualizations...")
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 2019 classification
        im1 = axes[0].imshow(class_2019, cmap=self.cmap, vmin=0, vmax=4, interpolation='nearest')
        axes[0].set_title('Land Cover Classification 2019\n(Before Pandemic)', 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Plot 2024 classification
        im2 = axes[1].imshow(class_2024, cmap=self.cmap, vmin=0, vmax=4, interpolation='nearest')
        axes[1].set_title('Land Cover Classification 2024\n(After Pandemic)', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Plot change map
        change_cmap = ListedColormap(['white', '#E63946', '#457B9D', '#F4A261'])
        im3 = axes[2].imshow(change_map, cmap=change_cmap, vmin=0, vmax=3, interpolation='nearest')
        axes[2].set_title('Change Detection Map\n(2019 → 2024)', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Create legend for land cover
        legend_elements = [
            mpatches.Patch(facecolor=self.colors[i], label=LAND_COVER_CLASSES[i])
            for i in sorted(LAND_COVER_CLASSES.keys())
        ]
        axes[1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
                      fontsize=10, title='Land Cover Classes')
        
        # Create legend for changes
        change_legend = [
            mpatches.Patch(facecolor='white', edgecolor='black', label='No change'),
            mpatches.Patch(facecolor='#E63946', label='New built-up'),
            mpatches.Patch(facecolor='#457B9D', label='Lost built-up'),
            mpatches.Patch(facecolor='#F4A261', label='Other changes')
        ]
        axes[2].legend(handles=change_legend, loc='upper left', bbox_to_anchor=(1.05, 1),
                      fontsize=10, title='Change Types')
        
        plt.tight_layout()
        filepath = FIGURES_DIR / f'change_detection_{self.model_name}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        
        # Create area change bar plot
        self.plot_area_changes(area_changes)
    
    def plot_area_changes(self, area_changes):
        """Plot area changes as bar chart"""
        classes = list(area_changes.keys())
        changes_km2 = [area_changes[c]['change_km2'] for c in classes]
        changes_pct = [area_changes[c]['change_percent'] for c in classes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute area change
        colors_bar = [self.colors[i] for i in range(len(classes))]
        bars1 = ax1.barh(classes, changes_km2, color=colors_bar, edgecolor='black')
        ax1.set_xlabel('Area Change (km²)', fontsize=12)
        ax1.set_title('Absolute Area Change (2019 → 2024)', fontsize=13, fontweight='bold')
        ax1.axvline(0, color='black', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, changes_km2)):
            if val != 0:
                ax1.text(val, i, f' {val:.3f}', va='center', 
                        ha='left' if val > 0 else 'right', fontsize=10)
        
        # Percentage change
        bars2 = ax2.barh(classes, changes_pct, color=colors_bar, edgecolor='black')
        ax2.set_xlabel('Percentage Change (%)', fontsize=12)
        ax2.set_title('Relative Area Change (2019 → 2024)', fontsize=13, fontweight='bold')
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, changes_pct)):
            if val != 0:
                ax2.text(val, i, f' {val:.1f}%', va='center',
                        ha='left' if val > 0 else 'right', fontsize=10)
        
        plt.tight_layout()
        filepath = FIGURES_DIR / f'area_changes_{self.model_name}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
    
    def generate_report(self, area_changes):
        """Generate text report of changes"""
        print("\n" + "="*70)
        print("CHANGE DETECTION RESULTS")
        print("="*70)
        
        report = []
        report.append("\n" + "="*70)
        report.append("LAND COVER CHANGE ANALYSIS: TRYSIL REGION (2019 → 2024)")
        report.append("="*70)
        
        report.append("\nMETHOD:")
        report.append(f"  Model used: {self.model_name.upper()}")
        report.append("  Study period: Summer 2019 vs Summer 2024")
        report.append("  Data source: Sentinel-2 (10m resolution)")
        report.append("  Study area: Trysil mountain region")
        
        report.append("\n" + "-"*70)
        report.append("AREA CHANGES BY LAND COVER CLASS:")
        report.append("-"*70)
        
        df_data = []
        for class_name in LAND_COVER_CLASSES.values():
            stats = area_changes[class_name]
            df_data.append({
                'Land Cover': class_name,
                '2019 (km²)': f"{stats['area_2019_km2']:.3f}",
                '2024 (km²)': f"{stats['area_2024_km2']:.3f}",
                'Change (km²)': f"{stats['change_km2']:+.3f}",
                'Change (%)': f"{stats['change_percent']:+.1f}%"
            })
        
        df = pd.DataFrame(df_data)
        report.append("\n" + df.to_string(index=False))
        
        report.append("\n" + "-"*70)
        report.append("KEY FINDINGS:")
        report.append("-"*70)
        
        # Analyze Built_up changes
        built_up = area_changes['Built_up']
        report.append(f"\n1. CABIN/BUILT-UP AREA EXPANSION:")
        report.append(f"   • 2019: {built_up['area_2019_km2']:.3f} km²")
        report.append(f"   • 2024: {built_up['area_2024_km2']:.3f} km²")
        report.append(f"   • Change: {built_up['change_km2']:+.3f} km² ({built_up['change_percent']:+.1f}%)")
        
        if built_up['change_km2'] > 0:
            report.append(f"   The built-up area increased by {built_up['change_km2']:.3f} km²,")
            report.append(f"   representing a {built_up['change_percent']:.1f}% increase.")
        
        # Analyze forest changes
        forest = area_changes['Forest']
        report.append(f"\n2. FOREST COVER:")
        report.append(f"   • 2019: {forest['area_2019_km2']:.3f} km²")
        report.append(f"   • 2024: {forest['area_2024_km2']:.3f} km²")
        report.append(f"   • Change: {forest['change_km2']:+.3f} km² ({forest['change_percent']:+.1f}%)")
        
        # Analyze grassland changes
        grassland = area_changes['Grassland']
        report.append(f"\n3. GRASSLAND:")
        report.append(f"   • 2019: {grassland['area_2019_km2']:.3f} km²")
        report.append(f"   • 2024: {grassland['area_2024_km2']:.3f} km²")
        report.append(f"   • Change: {grassland['change_km2']:+.3f} km² ({grassland['change_percent']:+.1f}%)")
        
        report.append("\n" + "="*70)
        report.append("INTERPRETATION:")
        report.append("="*70)
        report.append("\nThis analysis quantifies land cover changes in the Trysil region")
        report.append("between 2019 (before pandemic) and 2024 (after pandemic).")
        report.append("\nThe results provide data-driven evidence for cabin expansion")
        report.append("monitoring and environmental impact assessment.")
        report.append("\n" + "="*70)
        
        report_text = '\n'.join(report)
        print(report_text)
        
        # Save report
        report_file = RESULTS_DIR / f'change_detection_report_{self.model_name}.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        
        # Save as JSON
        json_file = RESULTS_DIR / f'change_detection_results_{self.model_name}.json'
        with open(json_file, 'w') as f:
            json.dump(area_changes, f, indent=2)
        
    
    def run(self):
        """Run complete change detection pipeline"""
        # Load 2019 imagery
        features_2019, shape, profile_2019, nan_mask_2019 = self.load_and_preprocess_image(
            RAW_DATA_DIR / 'sentinel2_2019.tif'
        )
        
        # Classify 2019
        class_2019 = self.classify_image(features_2019, shape, nan_mask_2019)
        
        # Load 2024 imagery
        features_2024, shape, profile_2024, nan_mask_2024 = self.load_and_preprocess_image(
            RAW_DATA_DIR / 'sentinel2_2024.tif'
        )
        
        # Classify 2024
        class_2024 = self.classify_image(features_2024, shape, nan_mask_2024)
        
        # Detect changes
        change_map, area_changes = self.detect_changes(class_2019, class_2024, shape)
        
        # Visualize results
        self.visualize_results(class_2019, class_2024, change_map, area_changes)
        
        # Generate report
        self.generate_report(area_changes)
        
        print("\n" + "="*70)
        print("="*70)

def main():
    """Main function"""
    # Use Random Forest (best performing model)
    detector = ChangeDetector(model_name='random_forest')
    detector.run()

if __name__ == "__main__":
    main()
