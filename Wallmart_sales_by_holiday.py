import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("clean_data.csv")

holiday_sales = df.groupby('IsHoliday_x')['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(6,5))
sns.barplot(x='IsHoliday_x', y='Weekly_Sales', data=holiday_sales, palette='coolwarm')
plt.title('Average Weekly Sales: Holiday vs Non-Holiday')
plt.xlabel('Is Holiday?')
plt.ylabel('Average Weekly Sales')
plt.show()

