import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.cluster import KMeans
import seaborn as sns


df = pd.read_csv('/Users/pranalikulkarni/Python/insurance_dataset.csv')  # Assumes preprocessed merged file

df['startdate'] = pd.to_datetime(df['policy_start_date'], format='%m/%d/%y')
df['enddate'] = pd.to_datetime(df['policy_end_date'], format='%m/%d/%y')
df['policy_duration'] = (df['enddate'] - df['startdate'])
df['tenure_months']=(df['policy_duration']).dt.days//30

#print(df['tenure_months'])

summary = df.rename(columns ={
    'policy_id': 'CustomerID',
    'renewal_count' : 'frequency',
    'last_renewal_months_ago' : 'recency',
    'tenure_months' : 'T',
    'avg_premium_paid' : 'monetary_value'
})

summary = summary[summary['recency']<=summary['T']]
summary = summary[(summary['frequency']>0) | (summary['recency']==0)]
summary = summary[(summary['frequency'] > 0) & (summary['monetary_value'] > 0)]

bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(summary['frequency'], summary['monetary_value'])

#predict purchase
summary['predicted_purchase'] = bgf.conditional_expected_number_of_purchases_up_to_time (12, summary['frequency'], summary['recency'], summary['T'])
summary['predicted_avg_premium'] = ggf.conditional_expected_average_profit(summary['frequency'], summary['monetary_value'])

summary['cltv'] = summary['predicted_purchase'] * summary['predicted_avg_premium']

# Customer Segmentation

kmeans = KMeans (n_clusters=3, random_state=42)
summary['segment'] = kmeans.fit_predict(summary[['cltv']])

def label_segment(row):
    if row['cltv'] > 2000:
        return 'Platinum'
    elif row['cltv'] > 1000:
        return 'Gold'
    else:
        return 'Silver'
    
summary['segment_label'] = summary.apply(label_segment, axis=1)

#visualization

plt.figure(figsize=(10, 6))
sns.histplot(data=summary, x='cltv', hue='segment_label', kde=True)
plt.title("CLTV Distribution by Segment")
plt.xlabel("Predicted CLTV")
plt.ylabel("Policy Count")
plt.show()

# Waterfall-style bar chart
summary.groupby('segment_label').agg({
    'cltv': 'mean',
    'CustomerID': 'count',
    'T': 'mean'
}).rename(columns={'CustomerID': 'policy_count', 'T': 'avg_tenure'}).plot(kind='bar', figsize=(10, 6))
plt.title("Segment Summary: Avg CLTV, Policy Count, Avg Tenure")
plt.ylabel("Values")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

summary.to_csv("cltv_segmented_output.csv", index=False)
print("CLTV segmentation complete. Results saved to cltv_segmented_output.csv.")