#graph of actual/predicted , 4th task
import matplotlib.pyplot as plt
import os

def analyze_results(y_test, y_pred, outputs_dir):
    print("Analyzing results...")
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual Rings")
    plt.ylabel("Predicted Rings")
    plt.title("Actual vs Predicted")
    plt.savefig(os.path.join(outputs_dir, 'actual_vs_predicted.png'))
    print("Analysis graph saved as actual_vs_predicted.png.")
    plt.close()
