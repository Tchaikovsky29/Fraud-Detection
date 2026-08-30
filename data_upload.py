"""
Pulls the Fraudulent E-Commerce Transactions dataset from Kaggle,
sorts the large file chronologically, splits it into 3 roughly-equal
batches (simulating data arriving over time), converts everything to
Parquet, and uploads + commits each piece to lakeFS.

The small file (~25k rows) is treated as a fixed holdout evaluation
set and is uploaded separately -- it should never be read by the
training pipeline.
"""

import os
import kagglehub
import pandas as pd
import lakefs

DATE_COL = "Transaction Date"

repo = lakefs.Repository("fraud-detection")
branch = repo.branch("main")

def upload_and_commit(data, remote_path: str, commit_msg: str):
    branch.object(remote_path).upload(data, pre_sign = False)
    ref = branch.commit(message=commit_msg)
    print(f"Committed {remote_path} -> {ref.get_commit().id}")

dataset_path = kagglehub.dataset_download("shriyashjagtap/fraudulent-e-commerce-transactions")
print("Kaggle dataset downloaded to:", dataset_path)

LARGE_FILE = "Fraudulent_E-Commerce_Transaction_Data.csv"
SMALL_FILE = "Fraudulent_E-Commerce_Transaction_Data_2.csv"

large_df = pd.read_csv(os.path.join(dataset_path, LARGE_FILE), parse_dates=[DATE_COL])
holdout_df = pd.read_csv(os.path.join(dataset_path, SMALL_FILE), parse_dates=[DATE_COL])

print(f"Large file: {len(large_df)} rows, {large_df[DATE_COL].min()} -> {large_df[DATE_COL].max()}")
print(f"Small file: {len(holdout_df)} rows, {holdout_df[DATE_COL].min()} -> {holdout_df[DATE_COL].max()}")

if "Transaction ID" in large_df.columns:
    overlap = set(large_df["Transaction ID"]) & set(holdout_df["Transaction ID"])
    if overlap:
        print(f"WARNING: {len(overlap)} overlapping Transaction IDs between large and small files!")
    else:
        print("No overlap between large and small files -- safe to use small file as holdout.")

large_df = large_df.sort_values(DATE_COL).reset_index(drop=True)

n = len(large_df)
third = n // 3
batch_1 = large_df.iloc[:third]
batch_2 = large_df.iloc[third: 2 * third]
batch_3 = large_df.iloc[2 * third:]

for i, batch in enumerate([batch_1, batch_2, batch_3], start=1):
    print(f"Batch {i}: {len(batch)} rows, {batch[DATE_COL].min()} -> {batch[DATE_COL].max()}")

batch_1[DATE_COL] = batch_1[DATE_COL].astype("datetime64[us]")
batch_2[DATE_COL] = batch_2[DATE_COL].astype("datetime64[us]")
batch_3[DATE_COL] = batch_3[DATE_COL].astype("datetime64[us]")

# upload_and_commit(
#     batch_1.to_parquet(index=False),
#     "raw/batch-1.parquet",
#     "rewriting parquet with millisecond precision (batch 1)",
# )

# upload_and_commit(
#     batch_2.to_parquet(index=False),
#     "raw/batch-2.parquet",
#     f"raw: batch 2 ({batch_2[DATE_COL].min().date()} to {batch_2[DATE_COL].max().date()})",
# )

upload_and_commit(
    batch_3.to_parquet(index=False),
    "raw/batch-3.parquet",
    f"raw: batch 3 ({batch_3[DATE_COL].min().date()} to {batch_3[DATE_COL].max().date()})",
)

# holdout_df[DATE_COL] = holdout_df[DATE_COL].astype("datetime64[us]")
# print("holdout dtype after conversion:", holdout_df[DATE_COL].dtype)
# upload_and_commit(
#     holdout_df.to_parquet(index=False),
#     "holdout/eval.parquet",
#     "rewriting parquet with millisecond precision"
# )

print("Done.")