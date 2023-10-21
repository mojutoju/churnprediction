import numpy as np
import pandas as pd

# Define the number of samples (customers)
num_samples = 80000

# Create a random seed for reproducibility
np.random.seed(0)

# Generate customer IDs
customer_ids = range(1, num_samples + 1)

# Generate subscription duration (in months)
subscription_duration = np.random.randint(1, 36, num_samples)  # Random between 1 and 36 months

# Generate payment frequency (e.g., monthly, annually)
payment_frequency = np.random.choice(['monthly', 'annually'], num_samples)

# Generate usage frequency (e.g., daily, weekly)
usage_frequency = np.random.choice(['daily', 'weekly', 'monthly'], num_samples)

# Generate customer support interactions
customer_support_interactions = np.random.randint(0, 30, num_samples)  # Random between 0 and 30 interactions

# Generate churn labels (1 for churned, 0 for retained)
churn_labels = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])  # Simulating 20% churn rate

# Create a dictionary to hold the data
data = {
    'CustomerID': customer_ids,
    'SubscriptionDuration': subscription_duration,
    'PaymentFrequency': payment_frequency,
    'UsageFrequency': usage_frequency,
    'CustomerSupportInteractions': customer_support_interactions,
    'Churn': churn_labels
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the synthetic data to a CSV file
df.to_csv('synthetic_churn_data.csv', index=False)
