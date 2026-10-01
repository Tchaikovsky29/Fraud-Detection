"""
MCP server exposing the fraud detection project's scored transaction data
as queryable tools, backed by real SQL (DuckDB) over data fetched live
from MinIO (predictions) and lakeFS (batch-4 raw transactions) -- same
fetch logic as the Streamlit app, not a separately pre-computed file.

Run directly for a local stdio MCP server:
    python fraud_analyst_mcp.py
"""
import io
import json
import os
import time
from dotenv import load_dotenv
import boto3
import duckdb
import lakefs
import pandas as pd
from mcp.server.mcpserver import MCPServer

load_dotenv()
mcp = MCPServer("fraud-detector-analyst")

CACHE_TTL_SECONDS = 30
_cache = {"df": None, "fetched_at": 0}

FORBIDDEN_KEYWORDS = ["insert", "update", "delete", "drop", "alter", "attach", "copy", "pragma", "create"]


def fetch_all_predictions():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT_URL"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["MINIO_SECRET_ACCESS_KEY"],
    )
    bucket = os.environ["BUCKET_NAME"]

    paginator = s3.get_paginator("list_objects_v2")
    dfs = []
    for page in paginator.paginate(Bucket=bucket, Prefix="predictions/"):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            dfs.append(pd.read_parquet(io.BytesIO(body)))

    if not dfs:
        raise RuntimeError("No prediction files found under predictions/ -- did the consumer run?")
    return pd.concat(dfs, ignore_index=True)


def fetch_batch4():
    repo = lakefs.Repository(os.environ["LAKEFS_REPO_NAME"])
    obj = repo.branch("main").object("raw/batch-4.parquet")
    with obj.reader(pre_sign=False) as f:
        return pd.read_parquet(io.BytesIO(f.read()))


def get_joined_data():
    """Fetches fresh if the cache is stale, matching the Streamlit app's TTL."""
    now = time.time()
    if _cache["df"] is None or (now - _cache["fetched_at"]) > CACHE_TTL_SECONDS:
        predictions_df = fetch_all_predictions()
        batch4_df = fetch_batch4()
        df = predictions_df.merge(
            batch4_df, left_on="transaction_id", right_on="Transaction ID", how="left"
        )
        _cache["df"] = df
        _cache["fetched_at"] = now
    return _cache["df"]


def get_connection():
    df = get_joined_data()
    con = duckdb.connect(database=":memory:")
    con.register("transactions", df)  # registers the live DataFrame directly, no file needed
    return con


@mcp.tool()
def query_transactions(sql_query: str) -> str:
    """
    Run a read-only SQL SELECT query against the 'transactions' table
    (live-joined prediction + raw transaction data, refreshed at most
    every 30s). Columns include: transaction_id, transaction_date,
    is_fraud_predicted, fraud_probability, actual_is_fraud,
    top_shap_factors, "Transaction Amount", "Payment Method",
    "Product Category", "Customer Age", "Account Age Days", and more --
    run `DESCRIBE transactions` first if unsure of the exact schema.

    Only SELECT statements are permitted. Always include LIMIT for
    row-level (non-aggregate) queries.
    """
    normalized = sql_query.strip().lower()
    if not normalized.startswith("select") and not normalized.startswith("describe"):
        return json.dumps({"error": "Only SELECT/DESCRIBE queries are permitted."})
    if any(word in normalized for word in FORBIDDEN_KEYWORDS):
        return json.dumps({"error": "Query contains a disallowed keyword."})

    try:
        con = get_connection()
        result = con.execute(sql_query).fetchdf()
        con.close()
    except Exception as e:
        return json.dumps({"error": str(e)})

    return result.to_json(orient="records")


@mcp.tool()
def get_overall_metrics() -> str:
    """Returns accuracy, precision, recall, and the project's cost-weighted
    score (5:1 false-negative:false-positive weighting) on the current
    live-fetched data."""
    con = get_connection()
    df = con.execute("SELECT actual_is_fraud, is_fraud_predicted FROM transactions").fetchdf()
    con.close()

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

    y_true = df["actual_is_fraud"].astype(int)
    y_pred = df["is_fraud_predicted"].astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fn_rate = fn / (fn + tp) if (fn + tp) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    cost = fn_rate * 5.0 + fp_rate * 1.0

    return json.dumps({
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "fn_rate": fn_rate, "fp_rate": fp_rate, "cost": cost,
        "total_transactions": len(df),
    })


if __name__ == "__main__":
    mcp.run()