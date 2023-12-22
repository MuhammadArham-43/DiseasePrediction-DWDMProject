from sklearn.metrics import accuracy_score, f1_score, r2_score
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def calulate_metrics(
        ground_labels: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        print(f"ACCURACY SCORE: {accuracy_score(ground_labels, predictions):.3f}")
        print(f"F1 Score: {f1_score(ground_labels, predictions, average='weighted'):.3f}")
        print(f"R2 Score: {r2_score(ground_labels, predictions):.3f}")
    

def display_confusion_matrix(ground_labels, predictions, classes):
    conf_matrix = confusion_matrix(ground_labels, predictions)
    
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt=".0f", xticklabels=classes, yticklabels=classes, linewidths=.5, cbar_kws={"shrink": 0.75})
    # Increase the size of each square
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    
    plt.title("Confusion Matrix")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    # plt.tight_layout()
    plt.show()