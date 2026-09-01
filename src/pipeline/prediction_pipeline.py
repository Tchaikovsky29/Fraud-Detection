import os
import json
from typing import Dict
import xgboost as xgb
import numpy as np
import kserve

class FraudDetectorPredictor(kserve.Model):
    """
    Combines preprocessing (the fitted Spark pipeline) and inference
    (the champion XGBoost model) in one component. Loads the fitted pipeline,
    champion model, and explicit string mappings from MLflow.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.spark = None
        self.preprocessing_pipeline = None
        self.model = None
        self.threshold = 0.5
        self.categorical_mappings = {}
        self.ready = False
        self.load()

    def load(self):
        import dagshub
        import mlflow
        import mlflow.spark
        import mlflow.xgboost
        from mlflow.tracking import MlflowClient
        from src.utils.main_utils import get_spark_session, read_yaml_file
        from pyspark.sql.types import DoubleType, StringType, StructField, StructType

        dagshub.auth.add_app_token(os.environ["DAGSHUB_USER_TOKEN"])
        dagshub.init(
            repo_owner=os.environ["DAGSHUB_REPO_OWNER"],
            repo_name=os.environ["DAGSHUB_REPO_NAME"],
            mlflow=True,
        )

        client = MlflowClient()
        registered_model_name = os.environ.get("REGISTERED_MODEL_NAME")

        self.spark = get_spark_session("fraud-detector-serving")
        champion_version = client.get_model_version_by_alias(registered_model_name, "champion")
        self.model = mlflow.xgboost.load_model(f"models:/{registered_model_name}@champion")
        
        champion_run = client.get_run(champion_version.run_id)
        champion_parent = champion_run.data.tags.get("mlflow.parentRunId")
        champion_parent_run = client.get_run(champion_parent)
        self.threshold = float(champion_parent_run.data.params["threshold"])
        self.preprocessing_pipeline = mlflow.spark.load_model(f"runs:/{champion_parent}/pipeline_model")

        # --- Load Categorical Mappings Artifact ---
        try:
            mappings_path = client.download_artifacts(
                run_id=champion_parent, 
                path="transformation_metadata/categorical_mappings.json"
            )
            with open(mappings_path, "r") as f:
                self.categorical_mappings = json.load(f)
        except Exception as e:
            # Fallback empty if artifact is missing from legacy runs
            self.categorical_mappings = {}

        schema_config = read_yaml_file("config/schema.yaml")
        categorical_cols = schema_config["categorical_columns"]
        numeric_cols = schema_config["numeric_columns"]
        passthrough_cols = schema_config["passthrough_columns"]

        # Build schema to dynamically infer feature order from vector output
        fields = []
        for col in categorical_cols:
            fields.append(StructField(col, StringType(), True))
        for col in numeric_cols + passthrough_cols:
            fields.append(StructField(col, DoubleType(), True))

        dummy_schema = StructType(fields)
        dummy_df = self.spark.createDataFrame([], schema=dummy_schema)
        transformed_df = self.preprocessing_pipeline.transform(dummy_df)

        features_metadata = transformed_df.schema["features"].metadata
        ml_attrs = features_metadata.get("ml_attr", {}).get("attrs", {})

        collected_attrs = []
        for attr_type, attr_list in ml_attrs.items():
            collected_attrs.extend(attr_list)

        collected_attrs.sort(key=lambda attr: attr["idx"])
        raw_feature_names = [attr["name"] for attr in collected_attrs]

        self.feature_names = []
        for name in raw_feature_names:
            if name.startswith("scaled_numeric_features_"):
                idx = int(name.split("_")[-1])
                self.feature_names.append(numeric_cols[idx])
            else:
                self.feature_names.append(name)
        
        self.ready = True

    def preprocess(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType, FloatType

        instances = payload["instances"]
        df = self.spark.createDataFrame(instances)

        df = df.withColumn("Transaction Date", F.to_timestamp("Transaction Date"))
        df = df.withColumn("Transaction Day", F.dayofmonth("Transaction Date"))
        df = df.withColumn("Transaction DOW", (F.dayofweek("Transaction Date") + 5) % 7)
        df = df.withColumn("Transaction Month", F.month("Transaction Date"))
        df = df.withColumn(
            "Is Address Match", (F.col("Shipping Address") == F.col("Billing Address")).cast("int")
        )
        for field in df.schema.fields:
            type_name = field.dataType.typeName()
            if type_name in ("long", "short", "byte"):
                df = df.withColumn(field.name, F.col(field.name).cast(IntegerType()))
            elif type_name == "double":
                df = df.withColumn(field.name, F.col(field.name).cast(FloatType()))

        transformed = self.preprocessing_pipeline.transform(df).select("features")
        pdf = transformed.toPandas()
        features = np.array(pdf["features"].apply(lambda v: v.toArray()).tolist())
        return {
            "features": features.tolist(),
            "instances": instances
        }

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        instances = payload["instances"]
        X = np.array(payload["features"])

        proba = self.model.predict_proba(X)[:, 1]
        preds = (proba >= float(self.threshold)).astype(int)

        dmatrix = xgb.DMatrix(X)
        contribs = self.model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_vals = contribs[:, :-1]

        results = []
        for i in range(len(preds)):
            raw_instance = instances[i]
            feature_impacts = []

            # Parse date if present to resolve derived calendar features
            tx_date_str = raw_instance.get("Transaction Date")
            derived_month = None
            if tx_date_str:
                try:
                    derived_month = int(tx_date_str.split("-")[1])
                except Exception:
                    pass

            for j in range(len(self.feature_names)):
                feature_name = self.feature_names[j]
                shap_val = float(shap_vals[i][j])

                # 1. Handle One-Hot Encoded Features with standard/indexer naming formats
                if "_ohe_" in feature_name or "_idx_" in feature_name:
                    sep = "_ohe_" if "_ohe_" in feature_name else "_idx_"
                    parent_col, category_str = feature_name.split(sep)
                    
                    # Convert numeric category string (e.g. "0.0") to integer index if needed
                    try:
                        cat_idx = int(float(category_str))
                        if parent_col in self.categorical_mappings and cat_idx < len(self.categorical_mappings[parent_col]):
                            category_val = self.categorical_mappings[parent_col][cat_idx]
                        else:
                            category_val = category_str
                    except ValueError:
                        category_val = category_str

                    raw_input_str = str(raw_instance.get(parent_col, "")).lower()
                    raw_val = 1 if raw_input_str == str(category_val).lower() else 0
                    display_feature_name = f"{parent_col}: {category_val}"

                # 2. Handle Derived Date/Time Features
                elif feature_name == "Transaction Month":
                    raw_val = derived_month if derived_month is not None else raw_instance.get("Transaction Month", "N/A")
                    display_feature_name = feature_name

                # 3. Handle Standard Input Features
                else:
                    raw_val = raw_instance.get(feature_name, "N/A")
                    display_feature_name = feature_name

                feature_impacts.append({
                    "feature": display_feature_name,
                    "shap_value": shap_val,
                    "raw_value": raw_val
                })

            top_factors = sorted(feature_impacts, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]

            results.append({
                "is_fraud": bool(preds[i]),
                "fraud_probability": float(proba[i]),
                "top_shap_factors": top_factors
            })

        return {"predictions": results}


if __name__ == "__main__":
    model = FraudDetectorPredictor(name="fraud-detector")
    kserve.ModelServer().start([model])