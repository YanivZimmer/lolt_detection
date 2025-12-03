import pandas as pd

df = pd.read_json("lotl_detector/data.jsonl", lines=True, orient="records")

print(df.head())


mask = df["claude-sonnet-4-5"].apply(lambda x: x.get("predicted_label")) != df["_label"]
df_disagree = df[mask]
relevant_columns = ["claude-sonnet-4-5", "_label", "_attack_technique", "_source", "prompt"]
print(df_disagree[relevant_columns].head())
df_disagree[relevant_columns].to_csv("lotl_detector/data_disagree.csv")
