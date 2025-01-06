import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data_path):
    columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    data = pd.read_csv(data_path, header=None, names=columns)

    numeric_data = data.drop(columns=['Sex'])

    X = numeric_data.drop(columns=['Rings'])
    y = numeric_data['Rings']

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split completed.")
    return X_train, X_test, y_train, y_test
