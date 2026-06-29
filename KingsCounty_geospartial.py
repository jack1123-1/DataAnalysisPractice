import folium
from folium.plugins import HeatMap
import pandas as pd

"""
#Description
-When analysing a geographic distributions, a geospartial heat map comes in hand.
-In this case it serves as a powerful visualization of how real estate value is distributed across an area.
-By weighting geographical coordinates with house prices, you can instantly identify economic hotspots.

#Understanding the distribution
*Red/Deep Orange Hotspots
-These areas represent the highest concentrations of expensive real estate. 
-There is a prominent cluster in the Seattle city center, moving north through Shoreline, and a significant high-value corridor east of Lake Washington encompassing Bellevue, Kirkland, and Redmond.

#Green/Yellow Zones
-These indicate moderate price points, covering the vast majority of the suburban sprawl in Renton, Kent, and Federal Way.

#Blue/Faint Outliers
-These show areas with fewer sales or lower-priced inventory, typically found in the more rural eastern and southern fringes of the county.
"""

df = pd.read_csv("house_data.csv")

geo_data = df[["lat", "long", "price"]].dropna()

# Center map roughly around Seattle
m = folium.Map(location=[47.61, -122.33], zoom_start=10)

# Heatmap data: lat, long, weight
heat_data = [
    [row["lat"], row["long"], row["price"]]
    for _, row in geo_data.iterrows()
]

HeatMap(
    heat_data,
    radius=15,
    blur=20,
    max_zoom=1
).add_to(m)

m.save("seattle_price_heatmap.html")


#Strategic Data Insights
"""
#The "Lake Effect" and Location Premium
-The most intense heat (red zones) is concentrated around Lake Washington and the Puget Sound coastline. 
-In King County data, proximity to water and the urban core are the primary drivers of high price-per-square-foot metrics.

#Market Segmentation
*The Luxury Hub 
-The eastern side of Lake Washington (Bellevue/Kirkland) shows high-intensity heat even in suburban settings, likely due to the proximity of major tech employers.
*The Value Corridor
-South King County (Auburn, Puyallup) remains "cooler," indicating these are the primary markets for entry-level buyers or those seeking more land for a lower price.
"""

#Feature Selection for modeling
"""
-When building a predictive model for house prices (like Linear Regression or Random Forest), this heatmap confirms that Latitude (lat) and Longitude (long) are among the most important features. 
-The price is not just about house size; it is fundamentally tied to these specific coordinates.
"""
