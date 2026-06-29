import pandas as pd
import numpy as np

train = pd.read_csv("wallmart2/train.csv")
test = pd.read_csv("wallmart2/test.csv")
features = pd.read_csv("wallmart2/features.csv")
stores = pd.read_csv("wallmart2/stores.csv")

train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])
features["Date"] = pd.to_datetime(features["Date"])

#preserve all training data on merge
train_merged = pd.merge(train, features, on=['Store', 'Date'], how='left')
train_merged = pd.merge(train_merged, stores, on='Store', how='left')

markdown_columns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
train_merged[markdown_columns] = train_merged[markdown_columns].fillna(0)

#train_merged.to_csv("clean_data.csv", index=False)