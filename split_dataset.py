import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv("./data/kor_hate_origin.csv", encoding="utf-8")

# topic_idx의 비율에 맞게 split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, val_idx in split.split(df, df["label"]):
    df_train = df.loc[train_idx]
    df_val = df.loc[val_idx]

df_train.to_csv("./data/kor_hate_train.csv", index=False)
df_val.to_csv("./data/kor_hate_val.csv", index=False)