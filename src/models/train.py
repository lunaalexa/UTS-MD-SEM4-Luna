import mlflow
import mlflow.sklearn
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score,KFold


from config.config import (RANDOM_STATE, MLFLOW_TRACKING_URI, MLFLOW_EXP_PIPELINE,
                           ARTIFACT_PIPELINE_CLASS, ARTIFACT_PIPELINE_REG, ACCURACY_THRESHOLD, R2_THRESHOLD)
from src.utils.io import save_artifact 

def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXP_PIPELINE)

def train_reg_optuna(pipeline, X_train, y_train):
    def objective(trial):
        params = {'regressor__n_estimators': trial.suggest_int('regressor__n_estimators',100,300),
              'regressor__max_depth': trial.suggest_int('regressor__max_depth',5,20),
              'regressor__min_samples_split': trial.suggest_int('regressor__min_samples_split',2,10),
              'regressor__min_samples_leaf' : trial.suggest_int('regressor__min_samples_leaf', 1, 6),
              'regressor__random_state': RANDOM_STATE}
        pipeline.set_params(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        return scores.mean()

    print("Optimizing Random Forest Regressor...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    _setup_mlflow()
    with mlflow.start_run(run_name="RegressionRFOptimized") as run:
        pipeline.set_params(**study.best_params)
        pipeline.fit(X_train, y_train)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_r2", study.best_value)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        
        save_artifact(pipeline, ARTIFACT_PIPELINE_REG)
        
        print(f"Regression model trained. Run ID: {run.info.run_id}")
        return run.info.run_id

def train_classification(pipeline, X_train, y_train)->str:
    _setup_mlflow()
    with mlflow.start_run(run_name="ClassificationLGBM") as run:
        pipeline.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        save_artifact(pipeline, ARTIFACT_PIPELINE_CLASS)
        
        print(f"Classification model trained. Run ID: {run.info.run_id}")
        return run.info.run_id























def train_pipeline(pipeline, X_train, y_train) -> str:
    def objective_lr(trial):
        params = {
            'classifier__C': trial.suggest_float('classifier__C', 0.001, 100, log=True),
            'classifier__penalty': trial.suggest_categorical('classifier__penalty', ['l1', 'l2']),
            'classifier__solver': trial.suggest_categorical('classifier__solver', ['liblinear', 'saga']),
            'classifier__max_iter': trial.suggest_int('classifier__max_iter', 500, 2000),
        }
        
        if params['classifier__solver'] == 'liblinear' and params['classifier__penalty'] not in ['l1', 'l2']:
            params['classifier__penalty'] = 'l2'
        
        pipeline.set_params(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        return scores.mean()

        

    print("Optimizing Logistic Regression...")
    from optuna.samplers import TPESampler
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study_lr = optuna.create_study(direction='maximize', study_name='logistic_regression_optimization', sampler=sampler)
    study_lr.optimize(objective_lr, n_trials=30, show_progress_bar=True)
    
    _setup_mlflow(MLFLOW_EXP_PIPELINE)

    with mlflow.start_run() as run:
        #log parameter
        #model_params = pipeline.named_steps['classifier'].get_params()
        #mlflow.log_params(model_params)
        pipeline.set_params(**study_lr.best_params)

        #training
        pipeline.fit(X_train, y_train)
        
        mlflow.log_params(study_lr.best_params)
        mlflow.log_metric("cv_accuracy", study_lr.best_value)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        
        save_artifact(pipeline, ARTIFACT_PIPELINE)
        
        print(f"Pipeline trained. Run ID: {run.info.run_id}")
        return run.info.run_id