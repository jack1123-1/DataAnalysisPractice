import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Telco-Customer-Churn.csv")

#Time based analysis
"""
The goal is to find the exit window.
-This step answers the question of When do customers leave.
	-Not who leaves
	-Not why
	-But when churn risk is highest
-For this task we use a KDE(Kernel Density Estimate) plot.
	-A smooth version of a histogram
	-used to visualise the probability density of a continous variable
	-It shows where values are concentrated, not just counted.
	-It answers the question of where the customers are clustering along the timeline.
"""

sns.kdeplot(
	data=df,
	x="tenure",
	hue="Churn",
	fill=True,
	common_norm=False,
	alpha=0.5
)

plt.title("Tenure Distribution by Churn Status")
plt.xlabel("Tenure(Months)")
plt.ylabel("Density")

plt.show()

#Data Insights
"""
*Hight Early Attrition
-The orange curve for Churn=1 shows a massive peak between 0 and 10 months.
-This indicates that the vast majority of customers who leave do so very early in their life cycle.

*The "Safe Zone"
-As tenure increases, the orange density drops significantly.
-If a customer stays past 20-month mark, their likelihood of churning decreases substantially.

*Bimodal Distribution for Loyal Customer
-The blue curve for Churn=0 is "bimodal," meaning it has two peaks.
-One smaller peak is at the beginning(new customers who haven't left yet).
-A larger, significant peak is around 70 months, respresenting a core highly loyal, long-term customers.

*The intersection Point
-Around 20 months, the two curves cross. Before this point, a random customer is more likely to be a "churner".
-After this point, they are statistically more likely to be a "stayer".
"""

#Bussiness and Analytical Implications
"""
*The Critical First Year
-The business should focus heavily on "Onboarding" and the first 6–12 months of the customer experience. 
-Since that is where the orange peak is highest, interventions here will have the highest ROI.

*Lock-in Effect
-Once a customer reaches the 60+ month mark, they are very stable. The goal should be to move customers from the "Early Phase" (left side) to the "Loyalty Phase" (right side).

*Data Cleaning Note
-Notice the curves extend slightly below 0 and above 80; this is a mathematical artifact of the KDE smoothing process, not necessarily an indication of negative months in your raw data.
"""











