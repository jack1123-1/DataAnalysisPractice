import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
*Shows:
	-features that move together
	-features that are redundant
	-features that indirectly drive churn
*This prevents a classic mistake of thinking price causes churn, when actually a service choice causes the high price.
*Correlation answers; when this goes up, does that also go up or down?
*Uses a color scale (typically Pearson correlation coefficient) ranging from -1.0 to 1.0:
    -Deep Red (1.0): Perfect positive correlation (variables move together).
    -Deep Blue (-1.0): Perfect negative correlation (as one goes up, the other goes down).
    -Neutral/Grey (0.0): No linear relationship.
"""

df = pd.read_csv("Telco-Customer-Churn.csv")

df_corr = df.copy()

df_corr = pd.get_dummies(df_corr, drop_first=True)
numeric_df = df_corr.select_dtypes(include=["int64", "float64"])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
	corr_matrix,
	cmap="coolwarm",
	center=0,
	linewidths=0.5,
	cbar=True
	)

plt.title("Correlation Heatmap of Telco Features")
plt.show()

#Key Predictors of Churn
"""
1.Looking at the Churn row:
*Tenure(Light Blue)
-Shows a clear negative correlation.
-This aligns with tenure distribution plot, as a customer's tenure increases, their likelihood of churninng decreases.

*Total Charges(Light Blue):
-Also shows a negative correlation with churn.
-Likely because high total charges are a proxy for long term tenure.

*Monthly Charges(Light Orange)
-Shows a positive correlation.
-Higher monthly bills are associated with a higher probability of churning.
"""

#Multi-Collinearity(Feature Overlap)
"""
1.Analysts look for dark red spots off the diagonal to find redundant data:
*Tenure vs TotalCharges
-There is strong positive correlation(dark orange/red) between these two.
-This makes sense-the longer you stay, the more you have paid in total.

*Analytical Impact
-In predictive modelling, keeping both highly correlated features can sometimes lead to multi-colinearity.
-This can confuse certain algorithms. An analyst might choose to drop one or combine them.

*Weak or Non-Existent Relationships
-The "Unnamed: 0" column(likely ID or index) shows near zero correlation with everything else(grey).
-Confirms it is "noise" and should be dropped before training a machine learning model.
"""

#summary
"""
Based on this heatmap, if you want to reduce churn, your primary focus should be on Tenure and MonthlyCharges. These two features have the most "color" in their relationship with Churn, meaning they hold the most predictive power.
"""