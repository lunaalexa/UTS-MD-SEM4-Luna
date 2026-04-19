from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def build_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    numeric_preprocess = Pipeline([
        ("num_imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_preprocess = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="constant",fill_value="Unknown")),
        ("cat_encoder", OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numPreprocess", numeric_preprocess, num_features),
            ("catPreprocess", categorical_preprocess, cat_features),
        ],
        remainder="drop",
    )