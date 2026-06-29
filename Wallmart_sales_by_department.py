import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("clean_data.csv")

dept_sales = df.groupby('Dept')['Weekly_Sales'].sum().reset_index().sort_values('Weekly_Sales', ascending=False)

plt.figure(figsize=(14,6))
sns.barplot(x='Dept', y='Weekly_Sales', data=dept_sales.head(20), palette='viridis')
plt.title('Top 20 Departments by Total Sales')
plt.xlabel('Department')
plt.ylabel('Total Sales')
plt.show()
