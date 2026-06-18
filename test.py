"""
Fake News Detector - Test Set Evaluation
Runs comprehensive evaluation on held-out test set
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Import our model
from model import FakeNewsClassifier

os.makedirs('results', exist_ok=True)

print("\n" + "=" * 60)
print("🧪 FAKE NEWS DETECTOR - TEST SET EVALUATION")
print("=" * 60 + "\n")

# Load model
model = FakeNewsClassifier()
model.load('model/fake_news_model.pkl')
print("✅ Model loaded from: model/fake_news_model.pkl")

# Load test data
if os.path.exists('results/test_data.pkl'):
    test_data = pd.read_pickle('results/test_data.pkl')
    X_test = test_data['X_test'].values
    y_test = test_data['y_test'].values
    print(f"✅ Test data loaded: {len(X_test):,} samples")
else:
    print("❌ Test data not found. Run train.py first!")
    exit(1)

print(f"\n📊 Test Set Distribution:")
print(f"   Fake:  {sum(y_test == 1):,} | Real: {sum(y_test == 0):,}")

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ============================================================================
# METRICS
# ============================================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("📈 TEST SET METRICS")
print("=" * 60)

print(f"""
  ╔════════════════════════════════════════════╗
  ║  Metric          │  Score                  ║
  ╠════════════════════════════════════════════╣
  ║  Accuracy        │  {accuracy:.4f} ({accuracy*100:.2f}%)         ║
  ║  Precision       │  {precision:.4f}                ║
  ║  Recall          │  {recall:.4f}                ║
  ║  F1-Score        │  {f1:.4f}                ║
  ╚════════════════════════════════════════════╝
""")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real News', 'Fake News']))

# Confusion matrix breakdown
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  Confusion Matrix Breakdown:")
print(f"    True Negatives (Real→Real):  {tn:,}")
print(f"    False Positives (Real→Fake): {fp:,}")
print(f"    False Negatives (Fake→Real): {fn:,}")
print(f"    True Positives (Fake→Fake):  {tp:,}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 60)
print("📊 GENERATING TEST VISUALIZATIONS")
print("=" * 60)

plt.style.use('seaborn-v0_8-whitegrid')

fig = plt.figure(figsize=(18, 12))

# Row 1: Confusion Matrix + Normalized CM
ax1 = fig.add_subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1,
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
            annot_kws={'size': 16, 'fontweight': 'bold'})
ax1.set_title('Confusion Matrix\n(Raw Counts)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Predicted', fontsize=11)
ax1.set_ylabel('Actual', fontsize=11)

ax2 = fig.add_subplot(2, 3, 2)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
            annot_kws={'size': 14, 'fontweight': 'bold'}, vmin=0, vmax=1)
ax2.set_title('Confusion Matrix\n(Row Normalized %)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Predicted', fontsize=11)
ax2.set_ylabel('Actual', fontsize=11)

# Row 1 col 3: ROC Curve
ax3 = fig.add_subplot(2, 3, 3)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color='#c0392b', lw=3, label=f'AUC = {roc_auc:.4f}')
ax3.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax3.fill_between(fpr, tpr, alpha=0.25, color='#c0392b')
ax3.set_xlabel('False Positive Rate', fontsize=11)
ax3.set_ylabel('True Positive Rate', fontsize=11)
ax3.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11, loc='lower right')
ax3.grid(True, alpha=0.3)

# Row 2 col 1: Precision-Recall Curve
ax4 = fig.add_subplot(2, 3, 4)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
ax4.plot(recall_curve, precision_curve, color='#8e44ad', lw=3, label=f'AP = {ap:.4f}')
ax4.fill_between(recall_curve, precision_curve, alpha=0.2, color='#8e44ad')
ax4.set_xlabel('Recall', fontsize=11)
ax4.set_ylabel('Precision', fontsize=11)
ax4.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# Row 2 col 2: Metrics Comparison Bar
ax5 = fig.add_subplot(2, 3, 5)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
colors_bar = ['#27ae60', '#2980b9', '#8e44ad', '#e67e22', '#c0392b']
bars = ax5.bar(metrics_names, metrics_values, color=colors_bar, edgecolor='black', alpha=0.85)
ax5.set_ylim([0, 1.15])
ax5.set_ylabel('Score', fontsize=11)
ax5.set_title('All Test Metrics', fontsize=13, fontweight='bold')
for bar, val in zip(bars, metrics_values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
             f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

# Row 2 col 3: Prediction Confidence Distribution
ax6 = fig.add_subplot(2, 3, 6)
real_proba = y_proba[y_test == 0]
fake_proba = y_proba[y_test == 1]
ax6.hist(real_proba, bins=40, alpha=0.65, label='Real News', color='#3498db', edgecolor='black')
ax6.hist(fake_proba, bins=40, alpha=0.65, label='Fake News', color='#e74c3c', edgecolor='black')
ax6.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Decision Threshold (0.5)')
ax6.set_xlabel('Predicted Probability of Fake', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.suptitle('Fake News Detector - Test Set Evaluation Report',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/test_evaluation_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/test_evaluation_results.png")

# ============================================================================
# SAVE RESULTS TO FILE
# ============================================================================
results_text = f"""
FAKE NEWS DETECTOR - TEST SET EVALUATION REPORT
================================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

TEST SET: {len(X_test):,} samples
  - Real News: {sum(y_test == 0):,}
  - Fake News: {sum(y_test == 1):,}

METRICS:
  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)
  Precision:   {precision:.4f}
  Recall:      {recall:.4f}
  F1-Score:    {f1:.4f}
  ROC AUC:     {roc_auc:.4f}
  Avg Precision: {ap:.4f}

CONFUSION MATRIX:
  True Negatives (Real→Real):  {tn:,}
  False Positives (Real→Fake): {fp:,}
  False Negatives (Fake→Real): {fn:,}
  True Positives (Fake→Fake):  {tp:,}

CLASSIFICATION REPORT:
{classification_report(y_test, y_pred, target_names=['Real News', 'Fake News'])}
"""

with open('results/test_report.txt', 'w') as f:
    f.write(results_text)
print("  Saved: results/test_report.txt")

print("\n" + "=" * 60)
print("🎉 TEST EVALUATION COMPLETE!")
print("=" * 60)
print(f"\n  📊 Visualization: results/test_evaluation_results.png")
print(f"  📝 Report:         results/test_report.txt")
print(f"\n  Run 'streamlit run app.py' to launch the web interface!")