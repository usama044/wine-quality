import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models import Model
from mlflow.models.signature import infer_signature
import numpy as np
import joblib
import tempfile
import os
from pathlib import Path
from src.wine_quality_project.entity.config_entity import ModelEvaluationConfig
from src.wine_quality_project.constants import *
from src.wine_quality_project.utils.common import save_json
from dotenv import load_dotenv


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Load from .env using path from config
        resolved_env_path = Path(self.config.dotenv_path).resolve()
        print("Resolved .env path:", resolved_env_path)
        print("Exists?", resolved_env_path.exists())

        if not resolved_env_path.exists():
            raise FileNotFoundError(f".env file not found at {resolved_env_path}")

        load_dotenv(dotenv_path=resolved_env_path)

        username = os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('MLFLOW_TRACKING_PASSWORD')
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')

        print("✅ .env loaded from:", resolved_env_path)
        print("URI:", tracking_uri)
        print("USERNAME:", username)

        if not all([username, password, tracking_uri]):
            raise EnvironmentError("One or more MLflow environment variables are missing in .env")

        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        mlflow.set_tracking_uri(tracking_uri)

        with mlflow.start_run():
            pred_y = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, pred_y)
            scores = {'rmse': rmse, "mae": mae, 'r2': r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('r2', r2)

            signature = infer_signature(test_x, pred_y)

            # Bypass registry: save model manually and log as artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = f"{temp_dir}/model"
                mlflow.sklearn.save_model(sk_model=model, path=model_path, signature=signature)
                mlflow.log_artifacts(model_path, artifact_path='model')