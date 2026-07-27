import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("company_financials_clean.csv")

df = load_data()

st.title("Financial Dashboard: [YOUR COMPANY NAME]")

# 3 KPI Cards
st.subheader("Latest Period KPIs")
c1, c2, c3 = st.columns(3)
latest_sales = df['Sales'].iloc[-1]
latest_profit = df['Net_Profit'].iloc[-1]
latest_opm = df['OPM_Percent'].iloc[-1]

c1.metric("Sales", f"₹{latest_sales:,.0f}")
c2.metric("Net Profit", f"₹{latest_profit:,.0f}")
c3.metric("OPM %", f"{latest_opm:.2f}%")

st.divider()

# Charts
st.subheader("Financial Trends")

# Chart 1: Sales & Net Profit
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(df['Period'], df['Sales'], marker='o', label='Sales')
ax1.plot(df['Period'], df['Net_Profit'], marker='s', label='Net Profit')
ax1.set_title('Sales & Net Profit Trend')
plt.xticks(rotation=45)
ax1.legend()
st.pyplot(fig1)

# Chart 2: OPM Percent
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.barplot(x='Period', y='OPM_Percent', data=df, color='skyblue', ax=ax2)
ax2.set_title('OPM Percent per Period')
plt.xticks(rotation=45)
st.pyplot(fig2)

# Chart 3: Boxplot
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.boxplot(x='Profit_Trend', y='Net_Profit', data=df, ax=ax3)
ax3.set_title('Net Profit by Profit Trend')
st.pyplot(fig3)

st.divider()

# Modeling
st.subheader("Predictive Model Comparison")
st.write("Predicting 'Profit_Trend' based on Sales_Growth, OPM_Percent, Interest, and Other_Income.")

X = df[['Sales_Growth', 'OPM_Percent', 'Interest', 'Other_Income']]
y = df['Profit_Trend']

lr = LogisticRegression().fit(X, y)
dt = DecisionTreeClassifier(random_state=42).fit(X, y)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [accuracy_score(y, lr.predict(X)), accuracy_score(y, dt.predict(X))],
    'Limitation': ['Small sample size', 'Small sample size']
})

st.dataframe(comparison_df)