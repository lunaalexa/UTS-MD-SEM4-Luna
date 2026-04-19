import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
from config.config import MLFLOW_TRACKING_URI, TARGET_CLASS, TARGET_REG

def evaluate(model, X_train, y_train, X_val, y_val, run_id: str, task_type: str = "classification") -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    metrics = {}
    with mlflow.start_run(run_id=run_id):
        if task_type == "classification":
            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, average='weighted')
            
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("gap_accuracy", abs(train_acc - val_acc))
            
            metrics = {"train_acc": train_acc, "val_acc": val_acc, "f1": val_f1}
            print(f"[{TARGET_CLASS}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {abs(train_acc-val_acc):.4f}")
        
        else: # regression
            train_r2 = r2_score(y_train, train_preds)
            val_r2 = r2_score(y_val, val_preds)
            val_mae = mean_absolute_error(y_val, val_preds)
            
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("val_r2", val_r2)
            mlflow.log_metric("gap_r2", abs(train_r2 - val_r2))
            
            metrics = {"train_r2": train_r2, "val_r2": val_r2, "mae": val_mae}
            print(f"[{TARGET_REG}] Train R2: {train_r2:.4f} | val R2: {val_r2:.4f} | Gap: {abs(train_r2-val_r2):.4f}")

    return metrics