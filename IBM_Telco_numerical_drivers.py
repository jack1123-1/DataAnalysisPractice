import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

df = pd.read_csv("Telco-Customer-Churn.csv")

sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("MonthlyCharges vs Churn")
plt.xlabel("Churn")
plt.ylabel("MonthlyCharges")
plt.show()

#Key observations form the boxplot
"""
MedianDifference
-The median monthly charge for customers who churned is notably higher-around $80-compared to approximately $65 for those who stayed.
-This suggests that customers with higher bills are more likely to leave.

Interquartile Range(IQR)
-The box of churned customers is shifted upward.
-The bottom 25% of churned are paying more(roughly $55+) than the bottom 25% of retained customers(roughly $25+).

Data Spread
-The whiskers show that both groups span almost the entire price range(from ~$20 to ~$120).
-The density of the churned group is concentrated in the higher price brackets.

Variance
-The box for non-churned customers is taller, indicating more variance in what loyal customers pay.
-They are distributed more broadly across low, medium and high price points.
"""

#what this means for business strategy
"""
Based on this plot, mothly charges is likely a "strong feature" for a predictive model. 
If you were building a machine learning model to predict churn, this variable would be a top contender for importance.

Actionable Insights:
1.Price sensitivity
-Customers paying above $70-$80 seem to be in a dangerous zone for leaving.

2.Target Retention
-The company might consider offering discounts or loyalty rewards specifically to customers whose monthly charges whos monthly charges cross that $80 threshold.

3.Value Proposition
-Data analysts should investigate why these charges are high
-Is it beacuse of premium features they dont value or are they simply being overcharged compared to competitors?
"""

"""Statistical Testing"""
#Are churners actually paying more?

churn_yes = df[df["Churn"] == 1]["MonthlyCharges"]
churn_no = df[df["Churn"] == 0]["MonthlyCharges"]

print(churn_yes.shape, churn_no.shape)

"""
Here we perfom a t-test although it may be risky since MonthlyCharges is not normally distributed.
"""
t_stat, p_value = ttest_ind(churn_yes, churn_no, equal_var=False)
print("T-test p-value:", p_value)

#p < 0.05 -> The difference in changes is statistically significant
#p >= 0.05 -> Difference may be due to chance


"""
Here we perfom the Mann-Whitney U test which is better with non uniformly distributed data.
The Mann-Whitney U Test is median based and Robust
"""
u_stat, p_value = mannwhitneyu(churn_yes, churn_no, alternative="two-sided")
print("Mann-Whitney p-value:", p_value)