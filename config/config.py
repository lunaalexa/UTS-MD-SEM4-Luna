from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_ING_DIR = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_PIPELINE_CLASS = ARTIFACTS_DIR / "placementStatus_classification_pipeline.pkl"
ARTIFACT_PIPELINE_REG   = ARTIFACTS_DIR / "salary_regression_pipeline.pkl"

TARGET_CLASS = "placement_status"
TARGET_REG = "salary_package_lpa"
DROP_COLS = ["student_id"]

NUM_FEATURES = ['ssc_percentage', 'hsc_percentage', 'degree_percentage', 'cgpa', 'entrance_exam_score', 'technical_skill_score',
                          'soft_skill_score', 'internship_count', 'live_projects', 'work_experience_months', 'certifications', 'attendance_percentage',
                          'backlogs','total_academic_avg',  'total_skill_score','total_experience_count','is_eligible' ,'tech_soft_ratio', 'exp_intensity', 
                          'low_attendance', 'credential_score']
CAT_FEATURES = ['gender','extracurricular_activities','gpa_group']

RANDOM_STATE = 42
TEST_SIZE = 0.2

RF_PARAMS = {
    'n_estimators': 238,
    'max_depth': 5,
    'min_samples_split': 7,
    'min_samples_leaf' : 5,
    'random_state': RANDOM_STATE
}

MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXP_PIPELINE = "Placement and Salary Prediction Pipeline"
ACCURACY_THRESHOLD = 0.98
R2_THRESHOLD = 0.84