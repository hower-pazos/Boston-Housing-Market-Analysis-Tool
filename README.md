# Boston Housing Market Analysis Tool

## Overview

The **Boston Housing Market Analysis Tool** is a Python-based project designed to explore and predict housing prices in Boston suburbs. Using a dataset of socio-economic and structural features, this tool implements data preprocessing, exploratory data analysis (EDA), and linear regression modeling to understand the factors influencing housing prices and make predictions for new data.

## Dataset

The dataset used in this project is the Boston Housing Dataset, which contains the following features:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxide concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built before 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property tax rate per \$10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
- **LSTAT**: % lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000s (target variable)

## Features of the Project

### 1. **Data Preprocessing**

- **Handling Missing Values**: Missing values are filled with the mean of the respective columns.
- **Outlier Treatment**: Outliers are removed using the Interquartile Range (IQR) method.

### 2. **Exploratory Data Analysis (EDA)**

- Visualization of feature distributions (e.g., histogram of housing prices).
- Correlation heatmap to identify relationships between variables.
- Pair plots to explore relationships between key features and the target variable (`MEDV`).

### 3. **Feature Engineering**

- Highly correlated features are identified to prevent multicollinearity.
- The `RAD` column, which exhibited high correlation with `TAX`, was excluded from the model to enhance interpretability and performance.

### 4. **Model Training**

- A **Linear Regression** model was trained to predict housing prices based on the cleaned and scaled dataset.
- Data was split into training (80%) and testing (20%) subsets for evaluation.

### 5. **Model Evaluation**

The Linear Regression model achieved the following results:

- **Metrics Used**:
  - Mean Squared Error (MSE): 7.0835
  - Root Mean Squared Error (RMSE): 2.6615
  - R-squared (R²): 0.6673

These metrics indicate that the model explains approximately 66.73% of the variance in housing prices, with an average error of approximately $2,661 (in $1000s).

### 6. **Prediction**

- The model supports predictions for new data by scaling it to match the training data and using the trained regression model.

## Example Prediction

```python
new_data = pd.DataFrame({
    'CRIM': [0.03],  
    'ZN': [0],        
    'INDUS': [8.14],  
    'CHAS': [0],      
    'NOX': [0.5],    
    'RM': [6.2],     
    'AGE': [70],  
    'DIS': [4.5],    
    'TAX': [300],     
    'PTRATIO': [18],  
    'B': [390],       
    'LSTAT': [12],    
})

new_data_scaled = scaler.transform(new_data)
new_data_pred = model.predict(new_data_scaled)
print("Predicted Housing Price:", new_data_pred)
```

Output:

```
Predicted Housing Price: [22.08396154]
```
This corresponds to a predicted price of $22,080.


## Key Python Libraries Used

- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical computations.
- **Matplotlib** & **Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning tasks such as data splitting, scaling, and model building.


## Setup Instructions

Before running the notebook, install the required dependencies:

1. Ensure you have **Python 3.8 or later** installed.
2. Install dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt


## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/hower-pazos/Boston-Housing-Market-Analysis-Tool.git
   ```
2. Open the Jupyter Notebook file (`Boston_Housing_Market_Analysis.ipynb`).
3. Run the notebook to:
   - Preprocess the data.
   - Visualize the relationships between features.
   - Train the linear regression model.
   - Predict housing prices for new data.
4. Optionally, modify the `new_data` DataFrame to include your own input values and generate predictions.

## File Structure

- **Boston\_Housing\_Market\_Analysis.ipynb**: The main Jupyter Notebook containing the analysis and model training steps.
- **HousingData.csv**: The dataset used for training and evaluation.
- **README.md**: Documentation for the project.