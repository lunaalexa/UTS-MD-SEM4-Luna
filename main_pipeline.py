from config.config import (ACCURACY_THRESHOLD,R2_THRESHOLD, NUM_FEATURES, CAT_FEATURES)
from src.data.loader import (ingest_data, load_frame, split_features_target, split_train_test)
from src.pipelines.sklearn_pipeline import create_classification_pipeline, create_regression_pipeline 
from src.models.train import train_classification, train_reg_optuna
from src.models.evaluate import evaluate

def main():
    print("=" * 50)
    print("sklearn Pipeline")
    print("=" * 50)

    print("\nStep 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Load and Split")
    df = load_frame() 
    X, y_class, y_reg = split_features_target(df)

    print("\nStep 3: Build and Train Pipeline")
    print("\nTASK 1: CLASSIFICATION (Placement)")
    x_train_c, x_val_c, y_train_c, y_val_c = split_train_test(X, y_class)
    
    class_pipeline = create_classification_pipeline()
    run_id_c = train_classification(class_pipeline,x_train_c,y_train_c)

    metrics_c = evaluate(class_pipeline, x_train_c, y_train_c, x_val_c, y_val_c, run_id_c, task_type="classification")
    acc = metrics_c['val_acc']

    if acc >= ACCURACY_THRESHOLD:
        print(f"Result: Model APPROVED (Accuracy={acc:.3f} >= {ACCURACY_THRESHOLD})")
    else:
        print(f"Result: Model REJECTED (Accuracy={acc:.3f} < {ACCURACY_THRESHOLD})")


    print("\nTASK 2: REGRESSION (Salary Package)")
    x_train_r, x_val_r, y_train_r, y_val_r = split_train_test(X, y_reg)
    
    reg_pipeline = create_regression_pipeline()
    run_id_r = train_reg_optuna(reg_pipeline, x_train_r, y_train_r)

    metrics_r = evaluate(reg_pipeline, x_train_r, y_train_r, x_val_r, y_val_r, run_id_r, task_type="regression")
    r2 = metrics_r['val_r2']

    if r2 >= R2_THRESHOLD:
        print(f"Result: Model APPROVED (R2-Score={r2:.3f} >= {R2_THRESHOLD})")
    else:
        print(f"Result: Model REJECTED (R2-Score={r2:.3f} < {R2_THRESHOLD})")


if __name__ == "__main__":
    main()