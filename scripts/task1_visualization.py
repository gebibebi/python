import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualize_data(data_path, outputs_dir):
    columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    data = pd.read_csv(data_path, header=None, names=columns)

    data_sample = data.sample(frac=0.1, random_state=42)  # Используем 10% данных

    numeric_data = data_sample.drop(columns=['Sex'])

    print("Step: Computing correlation matrix...")
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(outputs_dir, 'correlation_matrix.png'))
    print("Correlation matrix saved.")
    plt.close()

    print("Step: Creating scatter plots...")
    sns.pairplot(data_sample, diag_kind='hist', hue='Sex')  # Быстрый вариант графика
    plt.savefig(os.path.join(outputs_dir, 'scatter_plots.png'))
    print("Scatter plots saved.")
    plt.close()
