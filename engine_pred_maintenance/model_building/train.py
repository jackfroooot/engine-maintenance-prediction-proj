# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()
proj_name = "engine-maintenance-prediction-proj"

Xtrain_path = f"hf://datasets/jackfroooot/{proj_name}/Xtrain.csv"
Xtest_path  = f"hf://datasets/jackfroooot/{proj_name}/Xtest.csv"
ytrain_path = f"hf://datasets/jackfroooot/{proj_name}/ytrain.csv"
ytest_path  = f"hf://datasets/jackfroooot/{proj_name}/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define numeric and categorical features
numeric_features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[("num", "passthrough", numeric_features)    ],
    remainder="drop"
)
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.01, 0.05],
    'model__subsample': [0.7, 0.8],
    'model__colsample_bytree': [0.7, 0.8],
    'model__reg_lambda': [0.1, 1]
}
model_pipeline = Pipeline( steps=[("preprocess", preprocessor), ("model", xgb_model)] )
  # Using multiple scorings for log, while optimising for 'recall_fail'
scoring = {
    'recall_fail': make_scorer(recall_score, pos_label=0),
    'precision_fail': make_scorer(precision_score, pos_label=0),
    'f1_fail': make_scorer(f1_score, pos_label=0),
    'roc_auc': 'roc_auc'
}
with mlflow.start_run(run_name="XGBoost"):
   # Grid Search
   grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring=scoring, refit='recall_fail')
   grid_search.fit(Xtrain, ytrain)
   # Log parameter sets
   results = grid_search.cv_results_
   for i, params in enumerate(results['params']):
       with mlflow.start_run(nested=True):
           mlflow.log_params(params)
           mlflow.log_metrics({
               "cv_recall_fail": results['mean_test_recall_fail'][i],
               "cv_precision_fail": results['mean_test_precision_fail'][i],
               "cv_f1_fail": results['mean_test_f1_fail'][i],
               "cv_roc_auc": results['mean_test_roc_auc'][i]
           })
   # Best model
   mlflow.log_params(grid_search.best_params_)
   mlflow.log_metric("best_cv_recall_fail", grid_search.best_score_)
   best_model = grid_search.best_estimator_
   # Predictions
   y_pred_train = best_model.predict(Xtrain)
   y_pred_test  = best_model.predict(Xtest)
   y_proba_test = best_model.predict_proba(Xtest)[:, 1]

   # Metrics
   results_best_model = {
#       "model": "XGBoost",
       "train_recall_fail": recall_score(ytrain, y_pred_train, pos_label=0),
       "test_recall_fail": recall_score(ytest, y_pred_test, pos_label=0),
       "test_precision_fail": precision_score(ytest, y_pred_test, pos_label=0),
       "test_f1_fail": f1_score(ytest, y_pred_test, pos_label=0),
       "test_roc_auc": roc_auc_score(ytest, y_proba_test)
   }
   mlflow.set_tag("model", "XGBoost")
   mlflow.log_metrics(results_best_model)

     # Save the model locally
   model_path = "best_engine_maintenance_pred_model_v1.joblib"
   joblib.dump(best_model, model_path)

    # Log the model artifact
   mlflow.log_artifact(model_path, artifact_path="model")
   print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
   repo_id = f"jackfroooot/{proj_name}"
   repo_type = "model"

    # Step 1: Check if the space exists
   try:
       api.repo_info(repo_id=repo_id, repo_type=repo_type)
       print(f"Space '{repo_id}' already exists. Using it.")
   except RepositoryNotFoundError:
       print(f"Space '{repo_id}' not found. Creating new space...")
       create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
       print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
   api.upload_file(
       path_or_fileobj=model_path,
       path_in_repo=model_path,
       repo_id=repo_id,
       repo_type=repo_type,
   )
