import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load your large dataset
data = pd.read_csv('synthetic_churn_data.csv')

# Subsample the data
sampled_data = data.sample(n=10000)  # Adjust the number of samples as needed

# Create a histogram
plt.hist(sampled_data['SubscriptionDuration'], bins=20)
plt.xlabel('Subscription Duration (months)')
plt.ylabel('Frequency')
plt.title('Distribution of Subscription Duration (Sampled)')
plt.show()
