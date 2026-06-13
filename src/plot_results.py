import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config

def create_all_plots():
    RESULTS_DIR = os.path.join(config.BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Set global aesthetic theme for academic/project reports
    sns.set_theme(style="whitegrid", context="talk")

    print("📊 Generating Preprocessing Plots...")
    # ==========================================
    # 1. PREPROCESSING PLOTS
    # ==========================================
    prep_csv_path = os.path.join(RESULTS_DIR, 'preprocessing_mapping.csv')
    
    if os.path.exists(prep_csv_path):
        df_prep = pd.read_csv(prep_csv_path)
        
        # Plot 1A: Before vs After Label Count
        original_count = df_prep['Original Label'].nunique()
        canonical_count = df_prep['Canonical Class'].nunique()
        
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=['Original Variations', 'Canonical Classes'], 
                         y=[original_count, canonical_count], 
                         palette=['#e74c3c', '#2ecc71'])
        plt.title('Impact of Text Preprocessing on Class Count', fontweight='bold')
        plt.ylabel('Number of Classes')
        for i, v in enumerate([original_count, canonical_count]):
            ax.text(i, v + 2, str(v), ha='center', fontweight='bold', fontsize=14)
        
        plot1_path = os.path.join(RESULTS_DIR, 'plot_1_label_reduction.png')
        plt.savefig(plot1_path, bbox_inches='tight', dpi=300)
        plt.show()

        # Plot 1B: Top 10 Most Consolidated Classes
        variation_counts = df_prep.groupby('Canonical Class').size().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=variation_counts.values, y=variation_counts.index, palette='viridis')
        plt.title('Top 10 Classes with Most Linguistic Variations Merged', fontweight='bold')
        plt.xlabel('Number of Original Variations Mapped')
        plt.ylabel('Final Canonical Class')
        
        plot2_path = os.path.join(RESULTS_DIR, 'plot_2_top_consolidated.png')
        plt.savefig(plot2_path, bbox_inches='tight', dpi=300)
        plt.show()
    else:
        print("⚠️ Preprocessing CSV not found. Skipping preprocessing plots.")

    print("\n📈 Generating Evaluation Plots...")
    # ==========================================
    # 2. EVALUATION PLOTS
    # ==========================================
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    preds_path = os.path.join(RESULTS_DIR, 'predictions.csv')
    
    if os.path.exists(metrics_path) and os.path.exists(preds_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        df_preds = pd.read_csv(preds_path)

        # Plot 2A: Overall Metrics Bar Chart
        plt.figure(figsize=(8, 6))
        scores = [metrics['test_accuracy'], metrics['avg_bleu'], metrics['avg_chrf']]
        labels = ['Test Accuracy (%)', 'Avg BLEU Score', 'Avg CHRF Score']
        
        ax = sns.barplot(x=labels, y=scores, palette='Blues_d')
        plt.title('Overall Model Performance Metrics', fontweight='bold')
        plt.ylim(0, 100)
        for i, v in enumerate(scores):
            ax.text(i, v + 1, f"{v:.1f}", ha='center', fontweight='bold', fontsize=14)
            
        plot3_path = os.path.join(RESULTS_DIR, 'plot_3_overall_metrics.png')
        plt.savefig(plot3_path, bbox_inches='tight', dpi=300)
        plt.show()

        # Plot 2B: BLEU and CHRF Distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(df_preds['BLEU']*100, color="blue", label="BLEU", kde=True, stat="density", linewidth=0)
        sns.histplot(df_preds['CHRF']*100, color="orange", label="CHRF", kde=True, stat="density", linewidth=0)
        plt.title('Distribution of NLP Sentence Similarity Scores', fontweight='bold')
        plt.xlabel('Score (0-100)')
        plt.ylabel('Density')
        plt.legend()
        
        plot4_path = os.path.join(RESULTS_DIR, 'plot_4_score_distribution.png')
        plt.savefig(plot4_path, bbox_inches='tight', dpi=300)
        plt.show()

        # Extract Class-level F1-Scores
        clf_rep = metrics['classification_report']
        classes = []
        f1_scores = []
        for key, val in clf_rep.items():
            if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(val, dict):
                classes.append(key)
                f1_scores.append(val['f1-score'])
                
        df_f1 = pd.DataFrame({'Class': classes, 'F1': f1_scores})
        df_f1_sorted = df_f1.sort_values(by='F1', ascending=False)

        # Plot 2C: Top 10 Best Performing Classes
        top_10 = df_f1_sorted.head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='F1', y='Class', data=top_10, palette='Greens_r')
        plt.title('Top 10 Best Recognized Sentences (Highest F1-Score)', fontweight='bold')
        plt.xlabel('F1-Score')
        plt.xlim(0, 1.1)
        
        plot5_path = os.path.join(RESULTS_DIR, 'plot_5_best_classes.png')
        plt.savefig(plot5_path, bbox_inches='tight', dpi=300)
        plt.show()

        # Plot 2D: Top 10 Worst Performing Classes
        bottom_10 = df_f1_sorted.tail(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='F1', y='Class', data=bottom_10, palette='Reds_r')
        plt.title('Top 10 Most Confused Sentences (Lowest F1-Score)', fontweight='bold')
        plt.xlabel('F1-Score')
        plt.xlim(0, 1.1)
        
        plot6_path = os.path.join(RESULTS_DIR, 'plot_6_worst_classes.png')
        plt.savefig(plot6_path, bbox_inches='tight', dpi=300)
        plt.show()

        print(f"✅ All 6 high-quality plots saved successfully to: {RESULTS_DIR}")
    else:
        print("⚠️ Evaluation files (metrics.json or predictions.csv) not found. Run evaluate.py first.")

if __name__ == "__main__":
    create_all_plots()