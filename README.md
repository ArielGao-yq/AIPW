# AIPW
Example:
# Example Data
```
import pandas as pd
import numpy as np
features=['feature1', 'feature2','feature3']

num_rows = 100

np.random.seed(42)  # For reproducibility
feature1 = np.random.rand(num_rows) * 100  # Numerical values between 0 and 100
feature2 = np.random.rand(num_rows) * 50   # Numerical values between 0 and 50
feature3 = np.random.choice(['Category A', 'Category B', 'Category C'], size=num_rows)  # Categorical values

outcome = np.random.rand(num_rows) * 10  # Numerical values between 0 and 10
treatment = np.random.choice([0, 1], size=num_rows)  # Binary treatment (0 or 1)

df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'feature3': feature3,
    'outcome': outcome,
    'treatment': treatment
})

def create_dummies_for_all_categorical(X):
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    X_with_dummies = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    return X_with_dummies

df=df.reset_index(drop=True)

X=create_dummies_for_all_categorical(df[features])
Y = pd.Series(df['outcome'])
T = pd.Series(df['treatment'])
```

# Fit model

```
AIPW=AIPW_learner(X,T,Y,CI_method="CLT",trim_level=0.01) # CI_method can also be set to 'bootstrap', which has longer execution time.
results=AIPW.fit(X,T,Y) #returns ATE and CI
```
# Diagnostic plots
```
AIPW.balance_plt(X,T,Y)
AIPW.overlap_plot(X,T,Y)
AIPW.trim_summary(X,T,Y) # return percentage of samples being treated and the propensity score range of the trimmed observations.
AIPW.trim_shap(X,T) #SHAP summary plots for explaining which observations are more likely to be trimmed. Need to specify a trim level >0 when fit the model.
AIPW.shap_plot(X,T,Y) # SHAP summary plots for each model.
p=AIPW.propensity_score(X,T,Y) #return p_score, AUC, SHAP values for the propensity score model
```
