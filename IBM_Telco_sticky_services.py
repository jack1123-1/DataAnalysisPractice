import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#Hypothesis
"""
Customers who use more add-on services are harder to leave.

Why?
More services = more switching cost
More integration into daily habits
More perceived value

The goal is to prove or disprove this with data.
"""

df = pd.read_csv("Telco-Customer-Churn.csv")

addon_services = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

for service in addon_services:
	df[service] = df[service].apply(
		lambda x: 1 if x == "Yes" else 0
		)

df["ServiceCount"] = df[addon_services].sum(axis=1)

service_churn = (df.groupby("ServiceCount")["Churn"].mean().reset_index())

service_churn["ChurnRate"] = service_churn["Churn"] * 100

plt.figure(figsize=(7, 5))

sns.pointplot(
    x="ServiceCount",
    y="ChurnRate",
    data=service_churn
)

plt.ylabel("Churn Rate (%)")
plt.xlabel("Number of Add-on Services")
plt.title("Churn Rate vs Service Count")

plt.show()

"""
-This line plot illustrates the relationship between the Number of Add-on Services a customer subscribes to and their Churn Rate (%). 
-In data analysis, this is a powerful way to visualize how product "stickiness"—often called a product's moat—affects customer retention.
"""

#Key Analytical Insights
"""
*The "One-Service" Spike
-There is significant peak in churn for customers with exactly 1 add-on service.
-This suggests that customers who only commit to a single extra feature are highly volatile and may be testing the service without finding full value.

*The stickiness Effect
-Beyond one service, there is a consistent, steep negative correlation between the number of services and churn.
-As the count increases from 1 to 5, the churn rate drops from 40% to under 10%.

*Optimal Retention Point:
-Customers with 5 add-on-services exhibit lowest churn rate in the entire dataset.
-This represents the most "loyal" segment likely because the service has become deeply intergrated into their daily lives or business.

*Anormaly at Zero
-Interestingly, customers with 0 services have a lower churn rate (approx. 22%) than those with 1 service. 
-This might indicate a segment of "basic" users who have very low expectations or costs and are therefore less likely to actively cancel compared to those who tried one premium feature and were disappointed.
"""

#Connecting the full Dataset
"""
This final plot provides the "missing link" to your earlier analysis:
    High Costs vs. Value: While your Box Plot and Heatmap showed that higher monthly charges lead to churn, this plot clarifies that high charges alone aren't the enemy. If those charges come from multiple services, the customer is actually less likely to leave.

    Fiber Optic Context: Your Bar Chart showed Fiber Optic users churn at 40%. This line plot suggests that if those Fiber users only have one service, they are at extreme risk. Cross-selling them a second or third add-on (like Online Security or Streaming) could be the key to retaining them.
"""

#Strategic Recomendation
"""
From a data-driven business perspective, the most effective retention strategy is bundling. Moving a customer from 1 service to 2 or 3 services appears to be the most effective way to "lock in" their loyalty and offset the churn risk associated with high monthly fiber optic bills.
"""

