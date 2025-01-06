# This project is hosted on GitHub
# Link: https://github.com/gebibebi/python

import sys
import os

# Добавляем папку scripts в путь для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Импорты из модулей проекта
from task1_visualization import visualize_data
from task2_split_data import split_data
from task3_train_model import train_model
from task4_predictions import make_predictions
from task4part2_analysis import analyze_results
from task4part3_bias import plot_bias_variance
from task5_evaluation import evaluate_model

# Пути к данным и выходным файлам
data_path = os.path.join('data', 'abalone.data')
outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)

if __name__ == "__main__":
    try:
        # Шаг 1: Визуализация данных
        print("Step 1: Visualizing data...")
        visualize_data(data_path, outputs_dir)
        print("Visualization completed successfully.\n")

        # Шаг 2: Разделение данных
        print("Step 2: Splitting data...")
        X_train, X_test, y_train, y_test = split_data(data_path)
        print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")
        print("Data split completed successfully.\n")

        # Шаг 3: Обучение модели
        print("Step 3: Training model...")
        model = train_model(X_train, y_train, outputs_dir)
        print("Model training completed successfully.\n")

        # Шаг 4: Предсказания
        print("Step 4: Making predictions...")
        y_pred = make_predictions(model, X_test, y_test, outputs_dir)
        print("Predictions completed successfully.\n")

        # Шаг 4.1: Построение графика Bias and Variance Curve
        print("Step 4.1: Plotting Bias and Variance Curve...")
        plot_bias_variance(y_test, y_pred, outputs_dir)
        print("Bias and Variance Curve plotted successfully.\n")

        # Шаг 4.2: Анализ результатов
        print("Step 4.2: Analyzing results...")
        analyze_results(y_test, y_pred, outputs_dir)
        print("Analysis completed successfully.\n")

        # Шаг 5: Оценка модели
        print("Step 5: Evaluating model...")
        evaluate_model(y_test, y_pred, outputs_dir)
        print("Evaluation completed successfully.\n")

        # Завершение работы
        print("All steps completed successfully. Check the 'outputs' folder for results.")
    except Exception as e:
        print(f"An error occurred: {e}")
