#table with prediction, 4th task
import pandas as pd
import os
import statsmodels.api as sm

def make_predictions(model, X_test, y_test, outputs_dir):
    print("Making predictions...")
    X_test_sm = sm.add_constant(X_test)

    y_pred = model.predict(X_test_sm)

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    predictions = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    predictions.to_csv(os.path.join(outputs_dir, 'predictions.csv'), index=False)
    print(f"Predictions saved to {os.path.join(outputs_dir, 'predictions.csv')}.")

    return y_pred
