import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from config.config import (DATA_RAW_DIR, DATA_ING_DIR, DROP_COLS, NUM_FEATURES, CAT_FEATURES, TARGET_CLASS, TARGET_REG,RANDOM_STATE, TEST_SIZE
)

def ingest_data():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ING_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_file = DATA_RAW_DIR / "B-datasetUTSMD.csv"
    
    if not raw_file.exists():
        raise FileNotFoundError(f"File {raw_file} not found! insert dataset to data/raw/")
        
    df = pd.read_csv(raw_file)
    assert not df.empty, "Dataset is empty"

    out_file = DATA_ING_DIR / "B-datasetUTSMD_ingested.csv"
    df.to_csv(out_file, index=False)
    print(f"Data ingested: {raw_file} → {out_file}")

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop student ID
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])
    
    # Total Academic Score
    academic_cols = ['ssc_percentage', 'hsc_percentage', 'degree_percentage']
    df['total_academic_avg'] = df[academic_cols].mean(axis=1)
    
    # Total Skill Score
    df['total_skill_score'] = df['technical_skill_score']+df['soft_skill_score']
    
    # Experience Level
    df['total_experience_count'] = df['internship_count']+df['live_projects']
    
    # Eligible or no based on backlogs
    df['is_eligible'] = (df['backlogs'] < 3).astype(int)
    
    # Technical-Soft Skill Ratio
    df['tech_soft_ratio'] = df['technical_skill_score']/(df['soft_skill_score'] +1)
    
    # GPA Category
    df['gpa_group'] = pd.cut(df['cgpa'], bins=[0, 5.5, 7.6, 8.7, 9.8], 
                              labels=['Low','Average', 'High', 'Elite'])
    df['gpa_group'] = df['gpa_group'].astype(str)

    # Experience intensity
    df["exp_intensity"] = df["work_experience_months"]/(df["internship_count"] +1)

    # Attendance
    df["low_attendance"] = (df["attendance_percentage"]<80).astype(int)

    # Activities beside academic
    extra_map = {'Yes':1, 'No':0, 'True':1, 'False':0}
    temp = df['extracurricular_activities'].map(extra_map).fillna(0)
    df['credential_score'] = df['certifications'] + temp

    return df

def load_frame() -> pd.DataFrame:
    path = DATA_ING_DIR / "B-datasetUTSMD_ingested.csv"
    df = pd.read_csv(path)
    df = feature_engineering(df)
    return df

def split_features_target(df: pd.DataFrame):
    X = df[NUM_FEATURES+CAT_FEATURES]
    y_class = df[TARGET_CLASS].astype(int)
    y_reg = df[TARGET_REG].astype(float)
    return X, y_class, y_reg


def split_train_test(X: pd.DataFrame, y: pd.Series):
    stratify_param = y if y.name == TARGET_CLASS else None
    
    return train_test_split(X, y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=stratify_param)