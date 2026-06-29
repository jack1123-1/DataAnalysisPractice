import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("clean_data.csv")

store_type_sales = df.groupby('Type')['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(6,5))
sns.barplot(x='Type', y='Weekly_Sales', data=store_type_sales, palette='Set2')
plt.title('Average Weekly Sales by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Average Weekly Sales')
plt.show()