import asyncio
import io
import json
import os
import threading
import concurrent.futures

import boto3
import lakefs
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

load_dotenv()
st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

GRAFANA_URL = (
    "http://localhost:8080/d/adg6dfv/fraud-detector-serving-metrics"
    "?from=now-6h&to=now&timezone=browser&refresh=auto&kiosk"
)
COST_PER_MISSED_FRAUD = 5.0
COST_PER_BLOCKED_LEGIT_CUSTOMER = 1.0
MCP_SERVER_SCRIPT = os.environ.get("MCP_SERVER_SCRIPT", "mcp_server.py")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


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

@st.cache_data(ttl=30)
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


# ---------- Persistent MCP + Groq bridge ----------
# Keeps the MCP server subprocess and session alive across Streamlit
# reruns, on a dedicated background thread with its own event loop --
# spawning a fresh subprocess per chat message would both add latency
# and defeat the MCP server's own data cache.
def mcp_tools_to_groq_format(mcp_tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": getattr(t, "description", None),
                # some MCP tool objects use `input_schema` (snake_case)
                # while others may expose `inputSchema` (camelCase). Try both.
                "parameters": getattr(t, "input_schema", None) or getattr(t, "inputSchema", None),
            },
        }
        for t in mcp_tools
    ]


class MCPAgentBridge:
    def __init__(self, mcp_server_script):
        self.mcp_server_script = mcp_server_script
        self.loop = None
        self.session = None
        self.groq_tools = None
        self._ready = threading.Event()
        self._stop_event = None
        self._startup_error = None
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        if not self._ready.wait(timeout=30):
            if self._startup_error is not None:
                raise RuntimeError(f"MCP server failed to start: {self._startup_error}") from self._startup_error
            raise RuntimeError("Timed out starting MCP server connection (no error captured -- possible hang).")

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._connect_and_hold())
        except Exception as e:
            self._startup_error = e
            self._ready.set()

    async def _connect_and_hold(self):
        server_params = StdioServerParameters(command="python", args=[self.mcp_server_script])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                tool_list = await session.list_tools()
                self.groq_tools = mcp_tools_to_groq_format(tool_list.tools)
                # Instantiate Groq client inside the MCP bridge background context
                try:
                    self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                except Exception:
                    self.groq_client = None
                self._stop_event = asyncio.Event()
                self._ready.set()
                await self._stop_event.wait()  # keeps subprocess + session alive until shutdown

    def ask(self, groq_client, messages):
        if self.loop is None or not getattr(self.loop, 'is_running', lambda: False)():
            err_msg = (
                "MCP bridge not ready: background event loop is not running. "
                "Please wait a moment and try again."
            )
            messages.append({"role": "assistant", "content": err_msg})
            return err_msg, messages

        future = asyncio.run_coroutine_threadsafe(self._ask_async(groq_client, messages), self.loop)
        try:
            result = future.result(timeout=90)
            return result
        except concurrent.futures.TimeoutError:
            # Attempt to cancel the pending coroutine and return a user-friendly error
            try:
                future.cancel()
            except Exception:
                pass
            err_msg = (
                "The assistant took too long to respond. "
                "Please try again or reduce the scope of your question."
            )
            messages.append({"role": "assistant", "content": err_msg})
            return err_msg, messages
        except Exception as e:
            try:
                future.cancel()
            except Exception:
                pass
            messages.append({"role": "assistant", "content": f"Assistant error: {e}"})
            return f"Assistant error: {e}", messages

    async def _ask_async(self, groq_client, messages):
        # _ask_async runs on the MCP bridge background event loop
        system_msg = {
            "role": "system",
            "content": (
                "You are a fraud analyst assistant. You have tools to query a real dataset of "
                "transactions scored by a fraud detection model. Always use the tools to ground "
                "your answers in actual data -- never guess or make up numbers. When discussing "
                "why a transaction was flagged, reference its top_shap_factors. After receiving "
                "tool results, always respond with a complete natural-language answer -- never "
                "return an empty response."
            ),
        }
        current_messages = [system_msg] + messages

        # Prefer Groq client instantiated inside the bridge (same thread/loop).
        local_groq = getattr(self, "groq_client", None) or groq_client
        while True:
            try:
                response = local_groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=current_messages,
                    tools=self.groq_tools,
                    tool_choice="auto",
                )
            except Exception:
                raise
            message = response.choices[0].message

            if not message.tool_calls:
                text = message.content or ""
                messages.append({"role": "assistant", "content": text})
                return text, messages

            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": t.id,
                        "type": t.type,
                        "function": {"name": t.function.name, "arguments": t.function.arguments},
                    }
                    for t in message.tool_calls
                ],
            }
            messages.append(assistant_msg)
            current_messages.append(assistant_msg)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                result = await self.session.call_tool(tool_name, arguments=tool_args)
                result_text = "".join(b.text for b in result.content if hasattr(b, "text"))

                tool_msg = {"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": result_text}
                messages.append(tool_msg)
                current_messages.append(tool_msg)


# ---------- UI ----------

st.title("Fraud Detection -- Live Demo")

if st.button("Refresh data"):
    st.cache_data.clear()

try:
    predictions_df = fetch_all_predictions()
    batch4_df = fetch_batch4()
    df = predictions_df.merge(
        predictions_df.merge(batch4_df, left_on="transaction_id", right_on="Transaction ID", how="left")
        if False else batch4_df,  # placeholder guard, see note below
        left_on="transaction_id", right_on="Transaction ID", how="left"
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

if "mcp_bridge" not in st.session_state:
    with st.spinner("Connecting to MCP server..."):
        st.session_state.mcp_bridge = MCPAgentBridge(MCP_SERVER_SCRIPT)
        st.session_state.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

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
        try:
            _, st.session_state.chat_messages = st.session_state.mcp_bridge.ask(
                st.session_state.groq_client, st.session_state.chat_messages
            )
        except Exception as e:
            st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
    st.rerun()