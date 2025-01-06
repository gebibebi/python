from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

def evaluate_model(y_test, y_pred, outputs_dir):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with open(os.path.join(outputs_dir, 'model_metrics.txt'), 'w') as f:
        f.write(f"MSE: {mse}\nRMSE: {rmse}\nMAE: {mae}\nR²: {r2}\n")
