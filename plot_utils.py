import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(losses, optimizer_name):
    plt.figure(figsize=(8,5))
    plt.plot(losses, label=f"Optimizer: {optimizer_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(acc_list, optimizer_name):
    plt.figure(figsize=(8,5))
    plt.plot(acc_list, label=f"Optimizer: {optimizer_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_optimizers(loss_dict):
    plt.figure(figsize=(15,8))
    for name, losses in loss_dict.items():
        plt.plot(losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Optimizer Comparison: Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy_bars(pred_dict, y_test):

    accuracies = {name: np.mean(preds == y_test) for name, preds in pred_dict.items()}

    names = list(accuracies.keys())
    values = list(accuracies.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values)

    for bar, acc in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(),
                 f"{acc:.3f}",
                 ha='center', va='bottom', fontsize=10)

    plt.ylabel("Accuracy")
    plt.title("Test Accuracy Comparison Across Optimizers")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

def plot_rmse_bars(pred_dict, y_test):

    rmses = {
        name: np.sqrt(np.mean((preds - y_test) ** 2))
        for name, preds in pred_dict.items()
    }

    names = list(rmses.keys())
    values = list(rmses.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values)

    # Annotate bars with RMSE values
    for bar, rmse in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height(),
                 f"{rmse:.4f}",
                 ha='center', va='bottom', fontsize=10)

    plt.ylabel("RMSE")
    plt.title("Test RMSE Comparison Across Optimizers")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()



