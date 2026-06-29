import pandas as pd

#change dates to datetime format
#change bathrooms to int
#change floors to int.

df = pd.read_csv("kc_house_data.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
df["floors"] = pd.to_numeric(df["floors"], errors="coerce")

df = df[df["bedrooms"] < 20]
df = df[~((df["bedrooms"] == 0) & (df["bathrooms"] > 2))]

#df.to_csv("house_data.csv")