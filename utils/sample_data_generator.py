import pandas as pd
import numpy as np

def generate_sales_data(n=500):
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.randint(1000, 10000, n) + np.sin(np.arange(n)/30)*500,
        'Units': np.random.randint(10, 200, n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'Product': np.random.choice(['ProductA', 'ProductB', 'ProductC'], n),
        'Customer_ID': np.random.randint(1000, 2000, n),
        'Discount': np.random.uniform(0, 0.3, n).round(2),
        'Cost': np.random.randint(500, 5000, n),
    })
    df['Profit'] = df['Sales'] - df['Cost']
    return df

def generate_churn_data(n=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        'CustomerID': range(1, n+1),
        'Age': np.random.randint(18, 70, n),
        'Tenure': np.random.randint(1, 60, n),
        'MonthlyCharges': np.random.uniform(20, 120, n).round(2),
        'TotalCharges': np.random.uniform(100, 5000, n).round(2),
        'NumProducts': np.random.randint(1, 5, n),
        'HasCreditCard': np.random.randint(0, 2, n),
        'IsActiveMember': np.random.randint(0, 2, n),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
        'SupportTickets': np.random.randint(0, 10, n),
        'Churn': np.random.choice([0, 1], n, p=[0.73, 0.27])
    })
    return df