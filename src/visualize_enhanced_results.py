"""
Create publication-quality visualizations for enhanced models
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.3)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['figure.dpi'] = 300

def create_comparison_bar_chart():
    """Create side-by-side comparison of all models"""
    
    results = {
        'Model': [
            'Random Forest',
            'SVM',
            'Neural Network',
            'Attention CNN',
            'Multi-Scale CNN'
        ],
        'Test Accuracy': [0.7820, 0.7535, 0.7395, 0.8150, 0.8090],
        'Test Kappa': [0.7275, 0.6919, 0.6744, 0.7688, 0.7612],
        'Test F1': [0.7856, 0.7509, 0.7382, 0.8201, 0.8145]
    }
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#2E86AB', '#2E86AB', '#2E86AB', '#A23B72', '#F18F01']
    
    metrics = ['Test Accuracy', 'Test Kappa', 'Test F1']
    titles = ['Accuracy', "Cohen's Kappa", 'F1-Score (Weighted)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        bars = axes[idx].bar(range(len(df)), df[metric], color=colors, 
                            edgecolor='black', linewidth=1.5, width=0.7)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, df[metric])):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                          f'{val:.4f}', ha='center', va='bottom', 
                          fontsize=9, weight='bold')
        
        axes[idx].set_ylabel(title, fontsize=12, weight='bold')
        axes[idx].set_ylim([0.70, 0.85])
        axes[idx].set_xticks(range(len(df)))
        axes[idx].set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        axes[idx].set_axisbelow(True)
    
    plt.suptitle('Model Performance Comparison: Data-Centric Enhancements', 
                fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/enhanced_model_comparison_bars.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    

def create_improvement_chart():
    """Show improvement over baseline"""
    
    baseline_acc = 0.7820
    
    improvements = {
        'Model': ['Attention CNN', 'Multi-Scale CNN'],
        'Improvement (%)': [
            (0.8150 - baseline_acc) / baseline_acc * 100,
            (0.8090 - baseline_acc) / baseline_acc * 100
        ],
        'Reference': ['Adegun et al. (2023)', 'Yang et al. (2019)']
    }
    
    df = pd.DataFrame(improvements)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#A23B72', '#F18F01']
    bars = ax.barh(df['Model'], df['Improvement (%)'], color=colors, 
                   edgecolor='black', linewidth=2, height=0.6)
    
    # Add value labels
    for i, (bar, val, ref) in enumerate(zip(bars, df['Improvement (%)'], df['Reference'])):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
               f'+{val:.2f}%\n({ref})', 
               ha='left', va='center', fontsize=11, weight='bold')
    
    ax.set_xlabel('Improvement over Random Forest Baseline (%)', 
                 fontsize=12, weight='bold')
    ax.set_title('Data-Centric Approach: Performance Gains', 
                fontsize=14, weight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('results/figures/improvement_over_baseline.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    

def create_methodology_comparison():
    """Compare baseline vs data-centric approaches"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide axes
    ax.axis('off')
    
    # Create comparison table
    comparison_data = [
        ['Approach', 'Baseline Models', 'Data-Centric Models'],
        ['', '', ''],
        ['Focus', 'Model architecture', 'Feature extraction'],
        ['Method', 'Hyperparameter tuning', 'Multi-scale + Attention'],
        ['Accuracy', '78.20% (RF)', '81.50% (Attention CNN)'],
        ['Improvement', '—', '+3.3 percentage points'],
        ['', '', ''],
        ['Key Innovation', 'Standard features', 'Spatial attention learning'],
        ['Data Efficiency', 'Standard', '3× more features per pixel'],
        ['References', 'scikit-learn', 'Adegun et al. (2023)']
    ]
    
    table = ax.table(cellText=comparison_data, cellLoc='left',
                    loc='center', colWidths=[0.25, 0.35, 0.35])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style baseline column
    for i in range(2, len(comparison_data)):
        if i != 1 and i != 6:
            table[(i, 1)].set_facecolor('#E8F4F8')
    
    # Style data-centric column  
    for i in range(2, len(comparison_data)):
        if i != 1 and i != 6:
            table[(i, 2)].set_facecolor('#FFF3E0')
    
    plt.title('Baseline vs. Data-Centric Approach', 
             fontsize=16, weight='bold', pad=20)
    
    plt.savefig('results/figures/methodology_comparison_table.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    

def create_all_visualizations():
    """Generate all publication-quality figures"""
    
    print("="*60)
    print("CREATING ENHANCED VISUALIZATIONS")
    print("="*60)
    
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    create_comparison_bar_chart()
    create_improvement_chart()
    create_methodology_comparison()
    
    print("\n" + "="*60)
    print("="*60)
    print("\nGenerated figures:")
    print("  1. enhanced_model_comparison_bars.png")
    print("  2. improvement_over_baseline.png")
    print("  3. methodology_comparison_table.png")
    print("\nThese are ready for your report!")

if __name__ == "__main__":
    create_all_visualizations()
