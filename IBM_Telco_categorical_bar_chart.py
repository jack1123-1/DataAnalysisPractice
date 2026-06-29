import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
Answer this question:
Within each product category, how many customers churn?”

Not how many churned in total but What is the churn rate inside each group?”

Imagine:
Fiber Optic has 3,000 customers
DSL has 1,000 customers

Even if churn rates were equal:
Fiber will show more churn just because it’s larger

So raw counts answer:
“Which group is bigger?”

But we want:
“Which group is more dangerous?”

Churn Rate= Churned Customers/Total Customers in Group
"""


df = pd.read_csv("Telco-Customer-Churn.csv")


churn_by_internet = (
    df.groupby("InternetService")["Churn"]
    .mean()
    .reset_index()
)

churn_by_internet["ChurnRate"] = churn_by_internet["Churn"] * 100

plt.figure(figsize=(7, 5))

sns.barplot(
    x="InternetService",
    y="ChurnRate",
    data=churn_by_internet
)

plt.ylabel("Churn Rate (%)")
plt.xlabel("Internet Service Type")
plt.title("Churn Rate by Internet Service")

plt.show()

#Key Observations
"""
*Fiber Optic Risk
-Customers with Fiber optic service have higher churn rate by far exceeding 40%.
-This is a significant outlier compared to the other service types

*DSL satbility
-Customers using DSL churn at a much lower rate, approximately 19%.

*Baseline Retention
-Those with No internet service (likely phone-only customers) have the lowest churn rate, under 10%.

*Feature Importance
-This plot, combined the previous Correlation Heatmap explains why high Monthly Charge correlate with Churn.
-Fiber optic is typically a premium, higher cost service.
-Its high churn rate suggests customers may not feel the increased speed justifies the higher price tag.
"""

#Synthesis of the Data (The "Big Picture")
"""
When looking at this bar chart alongside your other provided plots, a clear narrative for this Telco dataset emerges:
    The "Why": The Heatmap showed that Monthly Charges and Churn are positively correlated. This Bar Chart identifies Fiber optic as the likely source of those high charges and subsequent churn.

    The "When": The Tenure Distribution plot showed that churn happens very early (0–10 months).

    The "Who": The Box Plot confirmed that churners (Group 1) consistently pay higher median monthly fees than those who stay.
"""

#Analytical Next Steps
"""
The extreme churn rate for Fiber Optic suggests a technical or value-proposition issue rather than a random occurrence.

    Service Quality Audit: Investigate if Fiber Optic customers are experiencing more technical outages or if the setup process is frustrating for new users.

    Competitive Analysis: Check if competitors are offering Fiber Optic at a significantly lower "introductory rate," causing customers to switch after their initial contract ends.

    Predictive Modeling: You should include "Internet Service Type" as a key categorical feature in any machine learning model you build, as it is a powerful indicator of churn.
"""