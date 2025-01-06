import pandas as pd
import os
import statsmodels.api as sm

def make_predictions(model, X_test, outputs_dir):
    print("Making predictions...")
    X_test_sm = sm.add_constant(X_test)
    y_pred = model.predict(X_test_sm)

    # Сохранение предсказаний
    predictions = pd.DataFrame({'Actual': X_test.index, 'Predicted': y_pred})
    predictions.to_csv(os.path.join(outputs_dir, 'predictions.csv'), index=False)
    print("Predictions saved to predictions.csv.")
    return y_pred
