import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd

def load_model_and_data():
    try:
        model = keras.models.load_model('chest_xray_model.keras')
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print("Error: Model files not found. Run train_model.py first.")
        exit(1)
    
    temp_dir = './temp_3class_split'
    if not os.path.exists(os.path.join(temp_dir, 'test')):
        print("Error: Test data not found. Run train_model.py first.")
        exit(1)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(temp_dir, 'test'),
        target_size=(128, 128),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    return model, test_generator, class_names

print("Loading model and test data...")
model, test_generator, class_names = load_model_and_data()
num_classes = len(class_names)

print(f"Test set: {test_generator.samples} images")
print(f"Classes: {class_names}\n")

print("Generating predictions...")
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))

report = classification_report(y_true, y_pred_classes, target_names=class_names, digits=4, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report.csv')
print("Saved: classification_report.csv")

cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

print("\nPer-Class Accuracy:")
for i, class_name in enumerate(class_names):
    if cm[i].sum() > 0:
        class_acc = cm[i, i] / cm[i].sum()
        print(f"  {class_name:15s}: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0], square=True, linewidths=1)
axes[0].set_title('Confusion Matrix - Counts', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1], square=True, linewidths=1)
axes[1].set_title('Confusion Matrix - Percentages', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("\nSaved: confusion_matrix.png")

y_true_bin = label_binarize(y_true, classes=range(num_classes))

plt.figure(figsize=(8, 6))
roc_auc_scores = {}

print("\nROC-AUC Scores:")
for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_scores[class_name] = roc_auc
    plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC={roc_auc:.3f})')
    print(f"  {class_name:15s}: {roc_auc:.4f}")

plt.plot([0,1], [0,1], 'k--', linewidth=1.5, label='Random')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
print("\nSaved: roc_curves.png")

macro_auc = roc_auc_score(y_true_bin, y_pred_probs, average='macro')
weighted_auc = roc_auc_score(y_true_bin, y_pred_probs, average='weighted')

summary_data = []
for i, class_name in enumerate(class_names):
    summary_data.append({
        'Class': class_name,
        'Precision': report[class_name]['precision'],
        'Recall': report[class_name]['recall'],
        'F1-Score': report[class_name]['f1-score'],
        'AUC-ROC': roc_auc_scores[class_name],
        'Support': int(report[class_name]['support'])
    })

summary_df = pd.DataFrame(summary_data)
print("\nMetrics Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv('metrics_summary.csv', index=False)
print("\nSaved: metrics_summary.csv")

overall_acc = np.sum(y_true == y_pred_classes) / len(y_true)
macro_f1 = report['macro avg']['f1-score']
misclassified = len(y_true) - np.sum(y_true == y_pred_classes)

print("\n" + "="*60)
print("FINAL TEST RESULTS")
print("="*60)
print(f"Overall Accuracy:      {overall_acc:.4f}")
print(f"Macro-Average F1:      {macro_f1:.4f}")
print(f"Macro-Average AUC-ROC: {macro_auc:.4f}")
print(f"Weighted AUC-ROC:      {weighted_auc:.4f}")
print(f"Misclassified:         {misclassified}/{len(y_true)} ({100*misclassified/len(y_true):.2f}%)")
print("="*60)

print("\nGenerated files:")
print("  - classification_report.csv")
print("  - confusion_matrix.png")
print("  - roc_curves.png")
print("  - metrics_summary.csv")
print("\nEvaluation complete!")
