# =========================================
# Shared Utilities
# =========================================

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# -----------------------------------------
# 1. Cross Validation
# -----------------------------------------
def get_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -----------------------------------------
# 2. Custom transformations
# -----------------------------------------
def drop_columns(X):
    return X.drop(columns=['duration', 'default'])


def transform_pdays(X):
    X = X.copy()
    X['previous_contact'] = (X['pdays'] != -1).astype(int)
    X['pdays'] = X['pdays'].replace(-1, 0)
    return X


# -----------------------------------------
# 3. Full preprocessing pipeline
# -----------------------------------------
def get_preprocessing_steps():
    
    numeric_features = [
        'age', 'balance', 'campaign',
        'pdays', 'previous', 'previous_contact'
    ]

    categorical_features = [
        'job', 'marital', 'education',
        'housing', 'loan', 'contact',
        'month', 'poutcome', 'day'
    ]

    numeric_transformer = Pipeline([
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return [
        ('drop_columns', FunctionTransformer(drop_columns)),
        ('pdays_transform', FunctionTransformer(transform_pdays)),
        ('preprocessor', preprocessor)
    ]