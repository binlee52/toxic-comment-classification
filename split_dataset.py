import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv("./data/kor_hate_origin.csv", encoding="utf-8")

# label 비율에 맞게 train, test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)

for train_idx, test_idx in split.split(df, df["label"]):
    df_train = df.loc[train_idx]
    df_subset = df.loc[test_idx]

# label 비율에 맞게 val, test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in split.split(df_subset, df_subset["label"]):
    df_val = df.loc[val_idx]
    df_test = df.loc[test_idx]

df_train.to_csv("./data/kor_hate_train.csv", index=False)
df_val.to_csv("./data/kor_hate_val.csv", index=False)
df_test.to_csv("./data/kor_hate_test.csv", index=False)