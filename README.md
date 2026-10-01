# Fraud Detection — AI Risk Manager

An end-to-end MLOps system for e-commerce fraud detection, built for Razorpay /buildathon's **Track 02: AI Risk Manager**. The focus here is not novel modeling — it's a production-shaped pipeline around it: versioned data, gated evaluation, honest metrics, explainable predictions, and live monitoring.

> **TL;DR**: XGBoost fraud classifier, trained via a fully orchestrated Kubeflow Pipeline, versioned with lakeFS, tracked in MLflow with automatic champion/challenger promotion, served via KServe with real-time SHAP explanations, monitored with Prometheus/Grafana, evaluated against a true held-out set replayed through Kafka, watched for drift by a standalone microservice, and explorable through a Streamlit dashboard with an LLM analyst chat grounded in the actual prediction data.

---

## Results

| Metric | Value |
|---|---|
| Accuracy | 94% |
| Precision | 38% |
| Recall | 42% |
| Cost (rate-weighted, 5:1 FN:FP) | 0.18 |

Metrics are computed on a **true held-out set** — transactions the model never saw during training or hyperparameter tuning, can be found at notebooks/final_evaluation.ipynb. These can also be relayed in through kafka to simulate a real production environment. See [Evaluation Methodology](#evaluation-methodology) for why this distinction matters.

---

## Architecture
```mermaid
%%{init: {
  "flowchart": {
    "nodeSpacing": 35,
    "rankSpacing": 80,
    "curve": "basis"
  }
}}%%

flowchart TD

subgraph group_data["Versioned Data"]
  direction TB

  node_upload["Chronological ingestion<br/>Python ingestion<br/>[data_upload.py]"]
  node_lakefs[("lakeFS repository<br/>versioned data lake")]
  node_minio[("MinIO<br/>object storage")]

  node_upload -->|"commits batches"| node_lakefs
  node_lakefs -->|"backed by"| node_minio
end


subgraph group_training["Training & Promotion"]
  direction TB

  node_training_pipeline{{"Kubeflow training pipeline<br/>Kubeflow Pipeline"}}
  node_commit_retrieval["Commit pinning<br/>pipeline component"]
  node_data_prep["Validation &amp; cleaning<br/>pipeline components<br/>[data_validation.py]"]
  node_feature_build["Spark feature pipeline<br/>PySpark preprocessing"]
  node_schema["Feature schema<br/>schema contract<br/>[schema.yaml]"]
  node_model_trainer["XGBoost trainer<br/>training component<br/>[model_trainer.py]"]
  node_evaluation["Cost-based evaluation<br/>evaluation component"]
  node_promotion["Champion promotion<br/>registry pusher<br/>[model_pusher.py]"]
  node_mlflow[("MLflow / DagsHub<br/>experiment tracking &amp; registry")]

  node_training_pipeline -->|"orchestrates"| node_commit_retrieval
  node_commit_retrieval -->|"pinned data"| node_data_prep
  node_data_prep -->|"validated data"| node_feature_build
  node_schema -->|"feature roles"| node_feature_build
  node_feature_build -->|"fitted features"| node_model_trainer
  node_model_trainer -->|"candidate model"| node_evaluation
  node_evaluation -->|"winner only"| node_promotion
  node_evaluation -->|"logs runs and metrics"| node_mlflow
  node_promotion -->|"registers champion"| node_mlflow
end


subgraph group_serving["Serving &amp; Replay"]
  direction TB

  node_kafka_producer["Kafka replay producer<br/>[producer.py]"]
  node_kafka_consumer["Kafka replay consumer<br/>[consumer.py]"]
  node_kserve{{"KServe predictor<br/>inference service"}}

  node_kafka_producer -->|"replay traffic"| node_kafka_consumer
  node_kafka_consumer -->|"prediction requests"| node_kserve
end


subgraph group_observability["Monitoring &amp; Analyst UI"]
  direction TB

  node_prometheus_grafana[("Prometheus &amp; Grafana<br/>metrics monitoring")]
  node_drift_detector{{"Evidently drift service<br/>Kubernetes microservice<br/>[drift_detector.py]"}}
  node_analyst_ui["Streamlit analyst UI<br/>analyst application<br/>[app.py]"]

  node_prometheus_grafana -->|"operational views"| node_analyst_ui
  node_drift_detector -->|"drift status"| node_analyst_ui
end


node_minio ~~~ node_training_pipeline
node_mlflow ~~~ node_kafka_producer
node_kserve ~~~ node_prometheus_grafana


node_lakefs -->|"resolves commit"| node_commit_retrieval

node_mlflow -->|"loads champion artifacts"| node_kserve

node_kafka_consumer -->|"commits live batch"| node_lakefs

node_kserve -->|"serving metrics"| node_prometheus_grafana

node_lakefs -->|"training and current data"| node_drift_detector

node_drift_detector -->|"HTML drift reports"| node_mlflow

node_kserve -->|"scores and SHAP"| node_analyst_ui


click node_upload "https://github.com/tchaikovsky29/fraud-detection/blob/main/data_upload.py"
click node_lakefs "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/configuration/lakefs_connection.py"
click node_training_pipeline "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/pipeline/training_pipeline.py"
click node_commit_retrieval "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/components/commit_retrieval.py"
click node_data_prep "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/components/data_validation.py"
click node_feature_build "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/components/data_transformation.py"
click node_schema "https://github.com/tchaikovsky29/fraud-detection/blob/main/config/schema.yaml"
click node_model_trainer "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/components/model_trainer.py"
click node_evaluation "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/components/model_evaluation.py"
click node_promotion "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/components/model_pusher.py"
click node_kserve "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/pipeline/inference-service.yaml"
click node_kafka_producer "https://github.com/tchaikovsky29/fraud-detection/blob/main/kafka_files/producer.py"
click node_kafka_consumer "https://github.com/tchaikovsky29/fraud-detection/blob/main/kafka_files/consumer.py"
click node_prometheus_grafana "https://github.com/tchaikovsky29/fraud-detection/blob/main/src/pipeline/kserve.pod_monitor.yaml"
click node_drift_detector "https://github.com/tchaikovsky29/fraud-detection/blob/main/drift-detection/drift_detector.py"
click node_analyst_ui "https://github.com/tchaikovsky29/fraud-detection/blob/main/app.py"


classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a

class node_upload,node_lakefs,node_minio toneBlue
class node_training_pipeline,node_commit_retrieval,node_data_prep,node_feature_build,node_schema,node_model_trainer,node_evaluation,node_mlflow,node_promotion toneAmber
class node_kserve,node_kafka_producer,node_kafka_consumer toneMint
class node_prometheus_grafana,node_drift_detector,node_analyst_ui toneRose
```
---
## Pipeline run (Kubeflow UI):
![Pipeline run](./screenshots/pipeline_run.png)

---

## Key Engineering Decisions

Documenting these explicitly, since the reasoning matters as much as the result:

- **Cost-weighted promotion, not accuracy.** On ~5% fraud prevalence, accuracy is trivially gamed by predicting "not fraud" for everything. Champion/challenger comparison uses a rate-based cost (`FN_rate × 5 + FP_rate × 1`) — normalized by class size so it stays comparable across differently-sized/balanced evaluation sets, not raw counts.
- **Decision threshold is swept, not fixed at 0.5.** `model_evaluation` searches candidate thresholds and selects the one minimizing cost, since `scale_pos_weight`-rebalanced models make the default 0.5 cutoff the wrong operating point.
- **Two-stage promotion gate.** `model_evaluation` checks an *absolute* quality floor independent of the current champion; `model_pusher` only then compares *relatively* against a freshly re-evaluated champion — on the **same test set**, not the champion's own historically-logged number, since (a) a model trained on more data will naturally look worse on its own harder test set even if more generalizable, and (b) a model performing well on historical data, but not on new data is not useful even if the training sizes are comparable
- **lakeFS for data lineage, not just storage.** Every training run resolves and pins an exact commit hash before running — cache correctness and reproducibility both depend on content-addressing the data, not trusting a branch name or a fixed path.
- **KServe predictor combines preprocessing and inference in one service**, rather than the standard predictor/transformer split — since preprocessing here is inherently tied to a fitted Spark ML pipeline, the split buys no flexibility, only complexity.
- **SHAP-based per-prediction explainability**, with feature names and category labels resolved back from the fitted pipeline's own metadata — not raw feature indices.
- **Honest holdout evaluation.** The reported metrics above come from replaying the true holdout set through the live serving path via Kafka, not from a train/test split reused during development.

### Disclosed deviations from the reference notebook
- `tree_method="hist"` (CPU) instead of the notebook's `"gpu_hist"` — no GPU in this environment.
- Feature engineering approach adapted from `https://www.kaggle.com/code/veerpatel6693/fraud-prediction`; the pipeline, versioning, evaluation gating, and serving infrastructure are original.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data versioning | lakeFS, MinIO |
| Processing | PySpark |
| Orchestration | Kubeflow Pipelines |
| Experiment tracking / registry | MLflow (via DagsHub) |
| Model | XGBoost |
| Serving | KServe, custom Python predictor |
| Streaming | Kafka |
| Monitoring | Prometheus, Grafana |
| Drift detection | Evidently |
| Demo UI | Streamlit |
| CI/CD | GitHub Actions |
| Infra | minikube, Docker |

---

## Data Versioning

Every raw batch and every training run's commit is tracked in lakeFS — full lineage from source data to trained model.

![lakeFS commit history](./screenshots/LakeFS_Commits.png)

---

## Experiment Tracking & Champion/Challenger Promotion

Every run — win or lose — is logged with full parameters and metrics in MLflow, nested under its parent pipeline run. Only models that beat the current champion (re-evaluated fresh, on the same test set) get promoted.

![MLflow run comparison](./screenshots/mlflow_runs.png)

---

## Live Serving & Monitoring

The deployed model is scored in real time via Kafka-replayed traffic, with request volume, latency, and the fraud/legitimate prediction split visible on a live Grafana dashboard.

![Grafana dashboard](./screenshots/grafana.png)

---

## Drift Detection

A standalone microservice periodically compares the champion's training data against a current data window using Evidently, exposing `dataset_drift_detected` and `drift_share` as Prometheus metrics and logging a full HTML report to MLflow per check.

An example report is included at [`eg_drift_report.html`](./eg_drift_report.html) — open it directly in a browser to see the full column-by-column drift breakdown.

---

## Demo: Explainable Fraud Analyst Chat

The Streamlit dashboard includes a tool-using LLM assistant grounded in the actual scored transaction data — it queries real prediction records and SHAP factors rather than answering from general knowledge.

![Assistant chat](./screenshots/assistant.png)

### LLM Assistant (MCP-backed)

- **Architecture**: The assistant runs as a Groq-hosted LLM that uses an MCP (Model Context Protocol) server process to expose safe, auditable tools (SQL queries, record lookups, SHAP retrievals) to the model. The MCP server is launched and held by the Streamlit app inside a background event loop so the model can call tools without blocking the UI.
- **Implementation notes**: The Streamlit `app.py` uses `MCPAgentBridge` to maintain a persistent MCP `ClientSession` and to instantiate the `Groq` client inside the bridge's background context. This ensures tool-calls and the Groq API run on the same thread/event-loop where the client was created, avoiding cross-thread issues.
- **Safety & grounding**: All analyst responses are grounded in actual data returned by MCP tools; the assistant is configured to never fabricate numeric facts and to reference top SHAP factors when explaining individual flags.


---

## Repository Structure

```
.
├── app.py                     # Streamlit demo: embedded Grafana, flagged transactions, LLM chat
├── data_upload.py             # One-time ingestion: Kaggle → chronological batches → lakeFS
├── eg_drift_report.html       # Example Evidently drift report
├── config/schema.yaml         # Column roles: categorical / numeric / passthrough / target
├── src/
│   ├── components/            # Kubeflow Pipeline components
│   │   ├── commit_retrieval.py
│   │   ├── data_validation.py
│   │   ├── data_cleaning.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── configuration/lakefs_connection.py
│   ├── entity/config_entity.py
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   ├── prediction_pipeline.py
│   │   ├── inference-service.yaml   # KServe InferenceService
│   │   └── kserve.pod_monitor.yaml  # Prometheus PodMonitor for the predictor
│   └── utils/main_utils.py
├── kafka_files/
│   ├── producer.py            # Replays holdout set as simulated live traffic
│   └── consumer.py            # Scores each row, logs predictions, commits batch-4.parquet
├── drift-detection/
│   ├── drift_detector.py      # Evidently-based drift microservice
│   ├── drift-detection.yaml   # Deployment + PodMonitor
│   └── Dockerfile
├── notebooks/
│   ├── fraud-prediction.ipynb         # Reference EDA + model comparison
│   └── hyperparameter_tuning.ipynb    # Optuna search
├── screenshots/                # README images
├── Dockerfile.base-env         # Base image: PySpark, lakeFS, no training deps
├── Dockerfile.inference        # Serving/training image: adds MLflow, XGBoost, KServe
├── docker-compose.lakefs.yml
├── docker-compose.kafka.yml
└── .github/workflows/          # CI: builds + pushes images on src/config changes
```

---

## Setup

**Prerequisites**: Docker, minikube, `kubectl`, Python 3.10, a Kaggle account, a DagsHub account.

1. **Local infra**: `docker compose -f docker-compose.lakefs.yml up -d` (lakeFS + MinIO), `docker compose -f docker-compose.kafka.yml up -d` (Kafka).
2. **Data ingestion**: `python data_upload.py` — pulls the dataset, splits chronologically, commits `batch-1` to lakeFS as the initial training set; holdout set committed separately.
3. **Cluster**: `minikube start`, install Kubeflow Pipelines, KServe, and `kube-prometheus-stack`.
4. **Secrets**: create `pipeline-secrets` (lakeFS, DagsHub, MinIO credentials) in every namespace that needs it — `kubeflow`, `kserve`, and wherever the drift detector runs.
5. **Images**: build and push via the included Dockerfiles, or let CI handle it on push to `main`.
6. **Pipeline**: `python src/pipeline/training_pipeline.py` to compile and upload, then trigger a run from the Kubeflow UI.
7. **Serving**: `kubectl apply -f src/pipeline/inference-service.yaml`.
8. **Monitoring**: `kubectl apply -f src/pipeline/kserve.pod_monitor.yaml` and `drift-detection/drift-detection.yaml`.
9. **Demo loop**: run `kafka_files/producer.py`, then `kafka_files/consumer.py`, then `demo.py` to compute final holdout metrics.
10. **Dashboard**: `streamlit run app.py`.

---

## Evaluation Methodology

Two distinct evaluation stages, intentionally separated:

1. **During training** (`model_evaluation`/`model_pusher`): scored against a train/test split of the *training* batch — used only for gating and promotion decisions, not reported as the headline metric.
2. **Final reported metrics** (this README, the demo): scored against a set held out since ingestion, never seen in any training run, when replayed through the actual deployed serving path via Kafka — as close to a genuine production evaluation as this environment allows.

**Known limitation**: once the holdout set is replayed, it's committed to lakeFS as `batch-4` and becomes eligible for future training. It remains a valid holdout for the *current* champion (a trained model has no memory of inference-time data — evaluating it repeatedly changes nothing), but a future retrain would need a freshly carved-out holdout to preserve this guarantee.

---

## License

See [LICENSE](./LICENSE).