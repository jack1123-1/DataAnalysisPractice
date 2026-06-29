import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Telco-Customer-Churn.csv")

"""Count Plt"""
sns.countplot(x="Churn", data=df)
plt.title("Churn distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

#from the plot i realise an imbalance in my bars with no churn dominating churn

"""Calculating the Baseline Churn Rate"""
total_customers = len(df)
total_churners = df["Churn"].value_counts()[1]

#churnrate = df["Churn"].mean()
churn_rate = total_churners/total_customers

print(f"Total Customers: {total_customers}")
print(f"Total Churners: {total_churners}")
print(f"Baseline Churn Rate: {churn_rate:.2%}")

#The baseline churn rate of 26%, it shows a model that will predict "No Churn" for everyone will be 74% accurate
#So the goal is to produce a model with accuracy greater than 74%

"""
NB: With data of this kind, it essentiaal to watch out for => 
	-Churn < 10% that shows severe imbalance and accuracy us misleading
	-Churn ~ 50% shows balance and accuracy is more meaningful
	-Churn > 40% shows High risk business problem

Confusion matric terms:
	| Term                | Meaning                       |
| ------------------- | ----------------------------- |
| True Positive (TP)  | Correctly predicted churn     |
| False Positive (FP) | Predicted churn, but didn’t   |
| False Negative (FN) | Missed a real churner         |
| True Negative (TN)  | Correctly predicted non-churn |


"""
