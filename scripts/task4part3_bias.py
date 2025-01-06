#❗graph bias/variance curve, 4th task also
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_bias_variance(y_test, y_pred, outputs_dir):
    bias = np.mean(np.abs(y_test - y_pred))

    variance = np.var(y_pred)

    errors = [bias, variance]
    labels = ['Bias', 'Variance']

    plt.bar(labels, errors, color=['blue', 'orange'])
    plt.title('Bias and Variance Curve')
    plt.xlabel('Error Component')
    plt.ylabel('Value')
    plt.savefig(os.path.join(outputs_dir, 'bias_and_variance_curve.png'))
    plt.show()
    print(f"Bias: {bias}, Variance: {variance}")
