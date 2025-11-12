"""
utils/evaluation.py
Fungsi-fungsi evaluasi model (print metrics, simpan confusion matrix)
"""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def print_metrics(y_true, y_pred):
    """Cetak accuracy dan classification report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

def plot_confusion(y_true, y_pred, labels=[0,1], save_path=None):
    """Gambar heatmap confusion matrix. Jika save_path diberikan simpan gambar."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
