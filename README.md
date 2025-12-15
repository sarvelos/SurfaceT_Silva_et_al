This repository contains all files necessary to reproduce the results of the manuscript:  
**“Machine-learning prediction of surface tension in ionic liquids using topological and quantum-informed molecular descriptors.”**

---

## Repository Contents

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `df_train_xgb_ready.csv` | Training set (preprocessed, with descriptors)     |
| `df_val_xgb_ready.csv`   | Validation set                                    |
| `df_test_xgb_ready.csv`  | Test set                                          |
| `final_model.joblib`     | Trained XGBoost model                             |
| `final_features.json`    | Feature list and scaling references               |
| `selected_features.txt`  | Final features used by the model                  |
| `IL_list_with_SMILES.csv`| IL names and corresponding SMILES                 |


## How to Use the Trained Model

```python
import pandas as pd
import joblib
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the trained model
model = joblib.load("final_model.joblib")

# Load feature list and scaling info (if applicable)
with open("final_features.json", "r") as f:
    features_info = json.load(f)

# Load selected features from TXT file
with open("selected_features.txt", "r") as f:
    selected_features = [line.strip() for line in f.readlines()]

# Load test set
df_test = pd.read_csv("df_test_xgb_ready.csv")

# Extract feature matrix in the correct order
X_test = df_test[selected_features]

# Predict in the same scale as training (log1p was applied to y)
y_pred_log = model.predict(X_test)

# Invert the log1p transformation to recover original scale (mN/m)
y_pred = np.expm1(y_pred_log)

# Ground truth values (already in original scale)
y_true = df_test["Surface_tension_liquid_gas_mN_m"].values

# Compute performance metrics
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Print results
print(f"MAE:  {mae:.2f} mN/m")
print(f"R²:   {r2:.3f}")

# Optionally attach predictions to the dataframe
df_test["Predicted_gamma"] = y_pred
```

---
## Data Provenance

All raw surface-tension data were sourced from the **NIST ILThermo 2.0** database [(DOI: 10.1021/je700171f)](https://doi.org/10.1021/je700171f).  
