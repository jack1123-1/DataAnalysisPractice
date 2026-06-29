import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house_data.csv")

df[["sqft_living", "sqft_lot", "price"]].corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Living area vs price
axes[0].scatter(df["sqft_living"], df["price"], alpha=0.3)
axes[0].set_xlabel("Living Area (sqft)")
axes[0].set_ylabel("Price")
axes[0].set_title("Price vs Living Area")

# Plot 2: Lot size vs price
axes[1].scatter(df["sqft_lot"], df["price"], alpha=0.3)
axes[1].set_xlabel("Lot Size (sqft)")
axes[1].set_ylabel("Price")
axes[1].set_title("Price vs Lot Size")

plt.tight_layout()
plt.show()

"""
1. Price vs. Living Area: The Linear Powerhouse

The left plot shows a clear positive linear correlation. As the sqft_living increases, the Price tends to increase alongside it.

    Strong Predictor: This is your primary candidate for a regression model. The tight grouping of data points suggests that "Living Area" has a high correlation coefficient (r).

    The "Fan" Effect (Heteroscedasticity): Notice how the points spread out as you move to the right. This is called heteroscedasticity, meaning your prediction error might be larger for massive mansions than for mid-sized family homes.

    Outlier Alert: Look at that lonely point far to the right (around 13,500 sqft) but at a relatively low price ($2M range). In a real analysis, you’d investigate if that’s a data entry error or perhaps a very large property in a less desirable area.

2. Price vs. Lot Size: The "Clump" Problem

The right plot is a classic example of a weak correlation.

    Vertical Stacking: Most of the data is squeezed against the left axis because most lots are relatively small, but their prices vary wildly (from $500k to $8M). This tells you that "Lot Size" alone is a poor predictor of value.

    Diminishing Returns: You can see properties with massive lots (over 1,000,000 sqft) that actually sell for less than small lots in the city center. This highlights why location (geospatial data) is more important than size.
"""