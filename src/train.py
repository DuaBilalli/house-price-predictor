import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from src.prepare import prepare

train_data, test_data = prepare()

cat_cols = train_data.select_dtypes(include="object").columns.tolist()
print(f"Kolonat kategorike: {len(cat_cols)}")

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([train_data[col], test_data[col]]).astype(str)
    le.fit(all_values)

    train_data[col] = le.transform(train_data[col].astype(str))
    test_data[col] = le.transform(test_data[col].astype(str))
    encoders[col] = le

print("\nEncoding is finished")

X = train_data.drop(columns=["SalePrice"])
y = np.log1p(train_data["SalePrice"])

print(f"\nFeatures: {X.shape} \nTarget: {y.shape}")

model = XGBRegressor(
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 4,
    random_state = 42,
    verbosity = 0
)

scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
rmse = -scores.mean()
print(f"\nRMSE mesatar: {rmse:.4f}")

model.fit(X, y)
print("\nTraining is finished")

joblib.dump({"model":model, "encoders":encoders}, "models/model.pkl")
print("\nModel is saved in: models/model.pkl")