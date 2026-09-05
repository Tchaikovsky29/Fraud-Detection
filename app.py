import io
import json
import os
from dotenv import load_dotenv
import boto3
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import lakefs
# Replaced anthropic with openai for OpenRouter
from openai import OpenAI

load_dotenv()
st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

GRAFANA_URL = (
    "http://localhost:8080/d/ad4dm2b/fraud-detector-serving-metrics"
    "?from=now-6h&to=now&timezone=browser&refresh=auto&kiosk"
)
COST_PER_MISSED_FRAUD = 5.0
COST_PER_BLOCKED_LEGIT_CUSTOMER = 1.0


# ---------- Data loading ----------

@st.cache_data(ttl=30)
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

st.cache_data(ttl=30)
def fetch_batch4():
    repo = lakefs.Repository(os.getenv("LAKEFS_REPO_NAME"))
    obj = repo.branch("main").object("raw/batch-4.parquet")
    with obj.reader(pre_sign=False) as f:
        return pd.read_parquet(io.BytesIO(f.read()))

def compute_metrics(df):
    y_true = df["actual_is_fraud"].astype(int)
    y_pred = df["is_fraud_predicted"].astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    cost = fn_rate * COST_PER_MISSED_FRAUD + fp_rate * COST_PER_BLOCKED_LEGIT_CUSTOMER
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "cost": cost}


# ---------- LLM tools, grounded in the actual dataframe ----------

def filter_transactions(df, is_fraud=None, min_probability=None, max_probability=None,
                         payment_method=None, limit=10):
    result = df
    if is_fraud is not None:
        result = result[result["is_fraud_predicted"] == is_fraud]
    if min_probability is not None:
        result = result[result["fraud_probability"] >= min_probability]
    if max_probability is not None:
        result = result[result["fraud_probability"] <= max_probability]
    if payment_method is not None:
        result = result[result["Payment Method"].str.lower() == payment_method.lower()]
    cols = ["transaction_id", "transaction_date", "Transaction Amount", "Payment Method",
            "Product Category", "fraud_probability", "actual_is_fraud", "top_shap_factors"]
    return result[cols].head(limit).to_dict(orient="records")


def get_summary_stats(df, group_by=None):
    if group_by and group_by in df.columns:
        grouped = df.groupby(group_by).agg(
            total=("is_fraud_predicted", "count"),
            flagged_fraud=("is_fraud_predicted", "sum"),
            avg_probability=("fraud_probability", "mean"),
        )
        grouped["fraud_rate"] = grouped["flagged_fraud"] / grouped["total"]
        return grouped.reset_index().to_dict(orient="records")
    return {
        "total_transactions": len(df),
        "flagged_fraud_count": int(df["is_fraud_predicted"].sum()),
        "fraud_rate": float(df["is_fraud_predicted"].mean()),
    }

# Translated to OpenAI/OpenRouter function schema format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "filter_transactions",
            "description": "Get individual transactions matching filters (fraud flag, probability range, payment method). Returns up to `limit` rows with full details including SHAP explanation factors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_fraud": {"type": "boolean", "description": "Filter to flagged (true) or not-flagged (false) transactions"},
                    "min_probability": {"type": "number"},
                    "max_probability": {"type": "number"},
                    "payment_method": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary_stats",
            "description": "Get aggregate fraud statistics, optionally grouped by a column (e.g. 'Payment Method', 'Product Category') to compare fraud rates across categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string", "description": "Column name to group by, e.g. 'Payment Method'"},
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_overall_metrics",
            "description": "Get overall model evaluation metrics (accuracy, precision, recall, cost) on this holdout data.",
            "parameters": {"type": "object", "properties": {}},
        }
    },
]


def run_tool(name, tool_input, df):
    if name == "filter_transactions":
        return filter_transactions(df, **tool_input)
    if name == "get_summary_stats":
        return get_summary_stats(df, **tool_input)
    if name == "get_overall_metrics":
        return compute_metrics(df)
    return {"error": f"unknown tool {name}"}


def chat_with_llm(df, messages):
    # Initialize OpenRouter Client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    
    system_msg = {
        "role": "system",
        "content": (
            "You are a fraud analyst assistant. You have tools to query a real dataset of "
            "transactions scored by a fraud detection model. Always use the tools to ground "
            "your answers in actual data -- never guess or make up numbers. When discussing "
            "why a transaction was flagged, reference its top_shap_factors."
        )
    }

    # Prepend the system prompt for OpenAI
    current_messages = [system_msg] + messages

    while True:
        response = client.chat.completions.create(
            model="meta/muse-spark-1.3-contributor",
            messages=current_messages,
            tools=TOOLS,
        )

        message = response.choices[0].message

        # If the LLM didn't call any tools, we're done
        if not message.tool_calls:
            text = message.content or ""
            messages.append({"role": "assistant", "content": text})
            return text, messages

        # Store the assistant's tool call request in history
        assistant_msg = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": t.id,
                    "type": t.type,
                    "function": {
                        "name": t.function.name,
                        "arguments": t.function.arguments
                    }
                } for t in message.tool_calls
            ]
        }
        messages.append(assistant_msg)
        current_messages.append(assistant_msg)

        # Execute all requested tools and append their results
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_input = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}
            
            result = run_tool(tool_name, tool_input, df)
            
            tool_result_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result, default=str),
            }
            messages.append(tool_result_msg)
            current_messages.append(tool_result_msg)


# ---------- UI ----------

st.title("Fraud Detection -- Live Demo")

if st.button("Refresh data"):
    st.cache_data.clear()

try:
    predictions_df = fetch_all_predictions()
    batch4_df = fetch_batch4()
    df = predictions_df.merge(
        batch4_df, left_on="transaction_id", right_on="Transaction ID", how="left"
    )
except Exception as e:
    st.error(f"Couldn't load data: {e}")
    st.stop()

metrics = compute_metrics(df)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Transactions scored", len(df))
m2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
m3.metric("Precision", f"{metrics['precision']:.3f}")
m4.metric("Recall", f"{metrics['recall']:.3f}")
m5.metric("Cost (rate-weighted)", f"{metrics['cost']:.3f}")

st.subheader("Live Serving Metrics")
st.components.v1.iframe(GRAFANA_URL, height=1000, scrolling=True)

st.subheader("Flagged Transactions")
flagged = df[df["is_fraud_predicted"] == True]
st.dataframe(
    flagged[["transaction_id", "transaction_date", "Transaction Amount", "Payment Method",
             "Product Category", "fraud_probability", "actual_is_fraud"]],
    use_container_width=True,
)

st.subheader("Ask about the data")
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Simplified message parsing for OpenAI message formats
for msg in st.session_state.chat_messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant" and msg.get("content"):
        with st.chat_message("assistant"):
            st.write(msg["content"])

user_input = st.chat_input("e.g. Why was transaction X flagged? Which payment method has the highest fraud rate?")
if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        _, st.session_state.chat_messages = chat_with_llm(df, st.session_state.chat_messages)
    st.rerun()