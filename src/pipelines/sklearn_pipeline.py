from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
from config.config import NUM_FEATURES, CAT_FEATURES, RANDOM_STATE
from src.features.pipeline_preprocessor import build_preprocessor

def create_classification_pipeline():
    preprocessor = build_preprocessor(NUM_FEATURES, CAT_FEATURES)
    pipeline = Pipeline([("preprocessor", preprocessor),("classifier", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1))])
    
    return pipeline

def create_regression_pipeline():
    preprocessor = build_preprocessor(NUM_FEATURES, CAT_FEATURES)
    pipeline = Pipeline([("preprocessor", preprocessor),("regressor", RandomForestRegressor(random_state=RANDOM_STATE))])
    
    return pipeline