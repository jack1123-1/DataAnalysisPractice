import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns      

#the goal is to see how sales cahnge week by week across all store
df = pd.read_csv("clean_data.csv")

weekly_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

plt.figure(figsize=(16,6))            
plt.plot(weekly_sales['Date'], weekly_sales['Weekly_Sales'], color='blue')
plt.title('Total Walmart Sales Over Time')
plt.xlabel('Date')                        
plt.ylabel('Weekly Sales')                 
plt.grid(True)              
plt.show()      
