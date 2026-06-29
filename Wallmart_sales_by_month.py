import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
A plot that focuses on time series seasonality.
"""

df = pd.read_csv("clean_data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
print(df.head())
monthly_sales = df.groupby("Month")['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(10,5))
sns.barplot(x='Month', y='Weekly_Sales', data=monthly_sales, palette='Blues_d')
plt.title('Average Sales per Month')
plt.xlabel('Month')
plt.ylabel('Average Weekly Sales')
plt.show()

#Seasonal trends and the holiday effect
"""
#The striking feature of the bar chart is the significant spinke in Months 11(November) and 12(December).

*Retail Seasonality
-A jump form ~15000 to ~20000 represents a massive surge in demand.
-This can be due to holidays like Black Friday and Christimas.

*Feature Engineering
-If one were to build a machine learning model to predict future sells, a binary feature called is_holiday would help the model predict a surge.
"""

#Stability Vs Volatility
"""
#Between Months 2 and 10, the average weekly sales are remarkably stable, hovering around the 15,000 to 16,000 range.

*Baseline Perfomance
-This establishes the baseline for optimum business.

*Anomalies
-Month 1 shows a slight deep which could be due to post holiday fatigue or missinf data points.
"""























