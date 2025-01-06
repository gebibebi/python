import statsmodels.api as sm
import os

def train_model(X_train, y_train, outputs_dir):
    print("Training the model...")
    X_train_sm = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_sm).fit()

    # Сохранение результатов модели
    with open(os.path.join(outputs_dir, 'regression_results.txt'), 'w') as f:
        f.write(model.summary().as_text())
    print("Model trained successfully and results saved.")
    return model
