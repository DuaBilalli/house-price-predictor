# Overview

This project focuses on predicting house prices using machine learning techniques applied to structured tabular data. The goal is to build a regression model that can accurately estimate the sale price of a house based on various features such as size, quality, location, and other property characteristics. The workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, and generating final predictions for unseen test data. Additionally, a Jupyter Notebook is used to perform visual analysis and better understand patterns in the dataset.

# Technologies

- Python
- Jupyter Notebook

# Libraries 

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

# Project Structure

```
house-price-predictor/
│
├── data/
│   ├── train-house.csv
│   ├── test-house.csv
│
├── src/
│   ├── prepare.py       
│   ├── train.py  
│   ├── predict.py 
│
├── notebooks/
│   ├── exploration.ipynb
│
├── models/
│   ├── model.pkl
│
├── submission.csv 
└── README.md
```

# Execution

Firstly, open your terminal and install all required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

Then, in the terminal, run the training script:
```
python src/train.py
```

Finally, generate predictions using:
```
python src/predict.py
```

# Model

The model used in this project is XGBoost Regressor, a high-performance gradient boosting algorithm widely used for regression problems. It works by building multiple decision trees sequentially, where each new tree tries to correct the errors of the previous ones. This makes it very effective for structured data like housing datasets. The model is trained on engineered features derived from the dataset, including both numerical and encoded categorical variables. A log transformation is applied to the target variable (SalePrice) to improve distribution stability and model performance. Cross-validation is used during training to ensure the model generalizes well and does not overfit.

# Results

After training, the model is able to generate predictions for the test dataset with reasonable accuracy. These predictions are exported into a submission.csv file, which follows the required format for Kaggle competition submission. The results show that the model captures important relationships between features and house prices, especially those related to overall quality, living area, and location. The final output can be directly uploaded to Kaggle for evaluation against the leaderboard.
