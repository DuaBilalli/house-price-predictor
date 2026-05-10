import pandas as pd

def prepare():
    train_data = pd.read_csv("data/train-house.csv")
    test_data = pd.read_csv("data/test-house.csv")

    print(f"Train: {train_data.shape[0]} rreshta, {train_data.shape[1]} kolona")
    print(f"Test: {test_data.shape[0]} rreshta, {test_data.shape[1]} kolona")

    cols_drop = ["PoolQC", "MiscFeature", "Alley", "Fence", "Id"]
    
    train_data = train_data.drop(columns=cols_drop)
    test_data = test_data.drop(columns=cols_drop)

    cat_fill = [
        "FireplaceQu", "GarageType", "GarageFinish", 
        "GarageQual", "GarageCond", "BsmtQual", 
        "BsmtCond", "BsmtExposure", "BsmtFinType1", 
        "BsmtFinType2", "MasVnrType"
    ]

    for col in cat_fill:
        if col in train_data.columns:
            train_data[col] = train_data[col].fillna("None")
            test_data[col] = test_data[col].fillna("None")

    num_fill = [
        "LotFrontage", "MasVnrArea", "GarageYrBlt"
    ]

    for col in num_fill:
        if col in train_data.columns:
            median_val = train_data[col].median()

            train_data[col] = train_data[col].fillna(median_val)
            test_data[col] = test_data[col].fillna(median_val)

    for col in train_data.columns:
        if train_data[col].isnull().sum() > 0:
            if train_data[col].dtype == "object":
                train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
                test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
            else:
                train_data[col] = train_data[col].fillna(0)
                test_data[col] = test_data[col].fillna(0)

    for col in test_data.columns:
        if test_data[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(test_data[col]):
                fill_val = train_data[col].median() if col in train_data.columns else 0
                test_data[col] = test_data[col].fillna(fill_val)
            else:
                fill_val = train_data[col].mode()[0] if col in train_data.columns else "None"
                test_data[col] = test_data[col].fillna(fill_val)

    print("\nPas pastrimit:")
    print(f"\nTrain: {train_data.isnull().sum().sum()}")
    print(f"\nTest: {test_data.isnull().sum().sum()}")

    return train_data, test_data

if __name__ == "__main__":
    train, test = prepare()
    print("\nData is cleaned")