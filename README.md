# **Abalone Prediction using Linear Regression**

🦪 This project predicts the age of abalone (a type of sea snail) based on physical measurements using a Linear Regression model. The dataset contains various features such as length, weight, and diameter of abalone, with the target variable being the number of rings, which corresponds to the age of the abalone.

---

## **Table of Contents**
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Workflow](#workflow)
5. [Outputs](#outputs)
6. [Technologies Used](#technologies-used)
7. [Author](#author)

---

## **Project Description**

The goal of this project is to train a Linear Regression model to predict the age of abalone based on physical measurements. The project involves data preprocessing, visualization, model training, evaluation, and analysis of results.

---

## **Dataset**
- **Source**: The Abalone dataset.
- **Attributes**:
  - `Sex`: Categorical feature (M, F, I for Male, Female, Infant).
  - `Length`, `Diameter`, `Height`, `Whole Weight`, `Shucked Weight`, `Viscera Weight`, `Shell Weight`: Continuous features.
  - `Rings`: Target variable (+1.5 gives the age in years).
- **Number of instances**: 4177.
- **Number of attributes**: 8.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/gebibebi/python.git
   cd python

2.Install required Python packages:

   pip install -r requirements.txt

****Workflow
**1. Data Visualization
Visualize the relationships between the features and the target variable (Rings).
Generate a correlation matrix to identify strong predictors.

**2. Data Preprocessing
Convert categorical variable Sex into numerical format.
Split the dataset into training and testing sets.
**3. Model Training
Train a Linear Regression model using the training data.
Present the intercept, coefficients, standard error, t-values, and p-values.
**4. Predictions
Predict the target variable on test data.

Generate:
A table of actual vs predicted values (predictions.csv).
Scatter plot of actual vs predicted (actual_vs_predicted.png).
Bias and variance curve (bias_and_variance_curve.png).
**5. Model Evaluation
Evaluate the model using metrics such as:
Mean Squared Error (MSE).
Root Mean Squared Error (RMSE).
Mean Absolute Error (MAE).
R-squared (R²).
**6. Analysis
Analyze and comment on the results with visualizations and metrics.
Outputs
The following outputs are saved in the outputs/ folder:

actual_vs_predicted.png: Scatter plot of actual vs predicted values.
bias_and_variance_curve.png: Bar chart showing bias and variance.
correlation_matrix.png: Heatmap of feature correlations.
model_metrics.txt: Evaluation metrics of the model.
predictions.csv: Contains actual and predicted values for the test set.
regression_results.txt: Intercept and coefficients of the trained model.
scatter_plots.png: Feature-wise scatter plots.

Technologies Used
Programming Language: Python 3.x

Libraries:
pandas for data manipulation.
matplotlib and seaborn for visualization.
statsmodels and sklearn for regression modeling and evaluation.

Author
Name: Dilnaz Shanova
Contact: GitHub
