#graph bias/variance curve, 4th task also
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_bias_variance(y_test, y_pred, outputs_dir):
    # Вычисляем Bias (средняя абсолютная ошибка)
    bias = np.mean(np.abs(y_test - y_pred))

    # Вычисляем Variance (дисперсия предсказаний)
    variance = np.var(y_pred)

    # Построение графика
    errors = [bias, variance]
    labels = ['Bias', 'Variance']

    plt.bar(labels, errors, color=['blue', 'orange'])
    plt.title('Bias and Variance Curve')
    plt.xlabel('Error Component')
    plt.ylabel('Value')
    plt.savefig(os.path.join(outputs_dir, 'bias_and_variance_curve.png'))
    plt.show()
    print(f"Bias: {bias}, Variance: {variance}")
