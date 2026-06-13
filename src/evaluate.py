import os
import json
import csv
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf

import config
from dataset import CSLDataset
from model import CSLModel

# Download NLTK data if not present
nltk.download('punkt', quiet=True)

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Evaluating on device: {device}")

    # --- 1. Setup Results Directory ---
    RESULTS_DIR = os.path.join(config.BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    report_path = os.path.join(RESULTS_DIR, 'evaluation_report.txt')
    json_path = os.path.join(RESULTS_DIR, 'metrics.json')
    csv_path = os.path.join(RESULTS_DIR, 'predictions.csv')
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')

    # --- 2. Load Data and Model ---
    test_dataset = CSLDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    idx_to_label = {v: k for k, v in test_dataset.label_map.items()}
    num_classes = len(idx_to_label)

    model = CSLModel(input_dim=config.FEATURE_DIM, num_classes=num_classes).to(device)
    
    # Load your specific v2 model
    model_path = os.path.join(config.MODELS_DIR, 'best_csl_model_v2.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    bleu_scores = []
    chrf_scores = []
    smoothie = SmoothingFunction().method1
    
    prediction_data = [] # For CSV

    print("Running predictions on the Test Set...\n")
    
    # --- 3. Inference Loop ---
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- 4. Calculate Scores & Format Output ---
    print("--- 📝 SAMPLE PREDICTIONS ---")
    for i in range(len(all_labels)):
        actual_sentence = idx_to_label[all_labels[i]]
        predicted_sentence = idx_to_label[all_preds[i]]
        
        # BLEU and CHRF
        ref_tokens = [actual_sentence.split()]
        hyp_tokens = predicted_sentence.split()
        b_score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
        c_score = sentence_chrf(actual_sentence, predicted_sentence)
        
        bleu_scores.append(b_score)
        chrf_scores.append(c_score)
        
        # Save to list for CSV
        match = "Yes" if actual_sentence == predicted_sentence else "No"
        prediction_data.append([actual_sentence, predicted_sentence, match, round(b_score, 4), round(c_score, 4)])
        
        # Print first 10 to Colab screen
        if i < 10:
            icon = "✅" if match == "Yes" else "❌"
            print(f"{icon} Actual: '{actual_sentence}' | Predicted: '{predicted_sentence}'")

    avg_bleu = np.mean(bleu_scores) * 100
    avg_chrf = np.mean(chrf_scores) * 100
    acc = accuracy_score(all_labels, all_preds) * 100
    
    clf_report_str = classification_report(all_labels, all_preds, zero_division=0)
    clf_report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)

    # --- 5. Print to Colab ---
    print("\n--- 📊 QUANTITATIVE RESULTS ---")
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Average BLEU Score: {avg_bleu:.2f}")
    print(f"Average CHRF Score: {avg_chrf:.2f}")
    print("\n--- CLASSIFICATION REPORT ---")
    print(clf_report_str)

    # --- 6. Save Text Report ---
    with open(report_path, 'w') as f:
        f.write("=== CSL Model Evaluation Report ===\n\n")
        f.write("--- Quantitative Results ---\n")
        f.write(f"Test Accuracy: {acc:.2f}%\n")
        f.write(f"Average BLEU Score: {avg_bleu:.2f}\n")
        f.write(f"Average CHRF Score: {avg_chrf:.2f}\n\n")
        f.write("--- Classification Report ---\n")
        f.write(clf_report_str)

    # --- 7. Save JSON Metrics ---
    metrics_dict = {
        "test_accuracy": acc,
        "avg_bleu": avg_bleu,
        "avg_chrf": avg_chrf,
        "classification_report": clf_report_dict
    }
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    # --- 8. Save CSV Predictions ---
    df_preds = pd.DataFrame(prediction_data, columns=["Actual", "Predicted", "Match", "BLEU", "CHRF"])
    df_preds.to_csv(csv_path, index=False)

    # --- 9. Plot and Save Confusion Matrix ---
    print(f"\nGenerating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(20, 20)) 
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=False)
    plt.title('Sentence Classification Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted Label ID', fontsize=16)
    plt.ylabel('True Label ID', fontsize=16)
    
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"✅ All results saved successfully in: {RESULTS_DIR}")
    
    # This command makes the plot show up in Google Colab!
    plt.show() 

if __name__ == "__main__":
    evaluate_model()