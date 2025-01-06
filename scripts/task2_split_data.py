import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data_path, test_size=0.2, random_state=42):
    # Загрузка данных
    columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    data = pd.read_csv(data_path, header=None, names=columns)

    # Удаляем колонку "Sex" и обрабатываем пропущенные значения (если есть)
    numeric_data = data.drop(columns=['Sex']).dropna()

    # Разделение на признаки (X) и целевую переменную (y)
    X = numeric_data.drop(columns=['Rings'])
    y = numeric_data['Rings']

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Логгирование размеров выборок
    print(f"Training set size: {len(X_train)} rows")
    print(f"Test set size: {len(X_test)} rows")
    
    print("Data split completed.")
    return X_train, X_test, y_train, y_test
