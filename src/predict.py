import sys 
import numpy as np 
import pandas as pd 
import joblib

sys.path.append("src")
from prepare import prepare

train, test = prepare()

saved = joblib.load("models/model.pkl")
model = saved['model']
encoders = saved['encoders']

for col, le in encoders.items():
    test[col] = le.transform(test[col].astype(str))

print("\nEncoding is finished")

test_X = test.drop(columns=["SalePrice"], errors="ignore")
log_predictions = model.predict(test_X)
predictions = np.expm1(log_predictions)

print(f"\nParashikime te gjeneruara: {len(predictions)}") 
print(f"Cmimi minimal: ${predictions.min():>10,.0f}") 
print(f"Cmimi maksimal: ${predictions.max():>10,.0f}") 
print(f"Cmimi mesatar: ${predictions.mean():>10,.0f}")

test_ids = pd.read_csv("data/test-house.csv")["Id"]
submission = pd.DataFrame({
    "Id": test_ids.values,
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)
print("submission.csv is saved")