# Financial Data Analysis & Predictive Modeling Report

## 1. Executive Summary

This report provides a comprehensive analysis of the company's financial performance data, spanning quarterly metrics (Mar 2023 – Mar 2026) and annual financial statements (FY 2015 – FY 2026). The project encompasses data preprocessing, exploratory data analysis (EDA), predictive machine learning modeling, and an interactive **Streamlit** dashboard for executive decision-making.

### Key Highlights
- **Revenue Growth:** Annual sales expanded from **₹31,972** in FY 2015 to **₹64,468** in FY 2026, representing a strong long-term upward trajectory.
- **Profitability:** Net Profit increased over threefold from **₹4,376** in FY 2015 to **₹15,059** in FY 2026, peaking alongside non-operating income spurts in recent periods.
- **Operating Margin Stability:** Operating Profit Margin (**OPM%**) has remained resilient within a narrow band of **17% to 25%**, averaging **~23%**.
- **Predictive Analytics:** Machine learning models (Logistic Regression and Decision Tree) were trained to forecast quarterly profit trends (`Profit Grew` vs. `Profit Declined`) based on sales growth, OPM%, interest, and other income.

---

## 2. Project Architecture & Structure

The repository comprises data processing pipelines, experimental Jupyter notebooks, cleaned data assets, and a web application interface:

```
d:\Aman_proj\
├── company_financials.csv        # Raw financial dataset (quarterly & annual)
├── company_financials_clean.csv  # Preprocessed & feature-engineered dataset
├── Untitled4.ipynb               # Data cleaning, EDA, and model experimentation notebook
├── app.py                        # Interactive Streamlit dashboard application
└── report.md                     # Comprehensive project report
```

### Technology Stack
- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`, `accuracy_score`)
- **Web Dashboard:** `streamlit`

---

## 3. Dataset & Data Cleaning Pipeline

### 3.1 Raw Dataset Overview
The raw dataset (`company_financials.csv`) contains 25 historical records covering the following features:
- `Period`: Financial period label (Quarterly e.g. *Mar-2023* and Annual e.g. *FY-Mar-2025*)
- `Sales`: Total revenue generated
- `Operating_Profit`: Profit generated from core business operations
- `OPM_Percent`: Operating Profit Margin percentage
- `Other_Income`: Non-operating revenue (investments, asset sales, etc.)
- `Interest`: Financial interest expenses
- `Depreciation`: Depreciation and amortization costs
- `Profit_Before_Tax`: Taxable profit
- `Net_Profit`: Bottom-line profit after tax
- `EPS`: Earnings Per Share

### 3.2 Data Preprocessing Steps
1. **Missing Value Handling:** Missing or null entries were interpolated using linear interpolation (`df.interpolate(method='linear', limit_direction='both')`).
2. **Growth Metrics Computation:**
   - $\text{Sales\_Growth} = \frac{\text{Sales}_t - \text{Sales}_{t-1}}{\text{Sales}_{t-1}}$
   - $\text{Net\_Profit\_Growth} = \frac{\text{Net\_Profit}_t - \text{Net\_Profit}_{t-1}}{\text{Net\_Profit}_{t-1}}$
3. **Target Label Creation (`Profit_Trend`):**
   - Categorized as `Profit Grew` if $\text{Net\_Profit\_Growth} > 0$
   - Categorized as `Profit Declined` if $\text{Net\_Profit\_Growth} \le 0$
4. **Data Sanitization:** Dropped initial rows containing `NaN` resulting from percentage change computations, yielding 24 clean observations saved to `company_financials_clean.csv`.

---

## 4. Exploratory Data Analysis (EDA) & Key Metrics

### 4.1 Statistical Summary (Clean Dataset, N=24)

| Metric | Sales (₹) | Net Profit (₹) | OPM % |
| :--- | :---: | :---: | :---: |
| **Mean** | 31,216.83 | 5,421.58 | 22.96% |
| **Median** | 24,243.00 | 4,263.50 | 24.00% |
| **Std Dev** | 18,102.07 | 3,461.71 | 2.12% |

### 4.2 Financial Performance Analysis
1. **Sales & Net Profit Co-movement:**
   - Sales exhibit steady quarterly performance (ranging between ₹15,190 and ₹16,514 across recent quarters) and annual expansion.
   - Significant net profit surges occurred in **Dec 2025** (₹6,603) and **FY 2026** (₹15,059), driven primarily by substantial spikes in **Other Income** (₹4,048 in Dec 2025; ₹4,923 in FY 2026).
2. **Operating Efficiency (OPM%):**
   - OPM% improved significantly from 17% in FY 2015 to stabilize around 23%–25% in recent periods, reflecting improved operational efficiency and pricing power.
3. **Profit Trend Distribution:**
   - Periods were evenly balanced between profit expansion and profit decline phases, providing a well-distributed binary classification target for machine learning.

---

## 5. Predictive Machine Learning Modeling

### 5.1 Problem Formulation
The goal is to predict whether net profit will grow or decline in a given period (`Profit_Trend`) based on financial indicators:
- **Feature Vector ($X$):** `Sales_Growth`, `OPM_Percent`, `Interest`, `Other_Income`
- **Target ($y$):** `Profit_Trend` (`Profit Grew` vs. `Profit Declined`)

### 5.2 Model Architecture & Evaluation
Two algorithms were trained and evaluated on the cleaned financial dataset:

| Model | Classification Accuracy | Key Strengths / Characteristics | Primary Limitation |
| :--- | :---: | :--- | :--- |
| **Logistic Regression** | **~75.0% - 83.3%** | Linear decision boundary, interpretable coefficients, robust against small noise | Sensitive to non-linear feature interactions |
| **Decision Tree Classifier** | **100.0%** (In-sample) | Captures non-linear decision boundaries and variable thresholds | High risk of overfitting due to small sample size ($N=24$) |

### 5.3 Model Insights & Limitations
- **Key Drivers:** `Other_Income` and `Sales_Growth` carry strong predictive signal for profit growth classification.
- **Sample Size Constraint:** With 24 observations, evaluating models purely on in-sample accuracy can lead to optimistic performance estimates for non-parametric models like Decision Trees. Cross-validation or expansion of historical quarterly records is recommended before production deployment.

---

## 6. Streamlit Interactive Dashboard (`app.py`)

The application provides an intuitive web interface for financial stakeholders to inspect metrics, trends, and model predictions in real time.

### Dashboard Features
1. **Executive KPI Header:** Displays the latest period metrics:
   - **Sales:** Latest period total revenue (e.g. ₹64,468)
   - **Net Profit:** Latest period net profit (e.g. ₹15,059)
   - **OPM %:** Latest operating profit margin percentage (e.g. 23.00%)
2. **Financial Trend Visualizations:**
   - **Sales & Net Profit Trend Line:** Dual line chart comparing revenue and profit over time.
   - **OPM% Bar Chart:** Historical breakdown of operating profit margins across periods.
   - **Profit Distribution Boxplot:** Categorical boxplot visualizing net profit distribution across growth vs. decline periods.
3. **Predictive Model Comparison Table:** Interactive table showing comparative performance metrics and limitations of the machine learning algorithms.

---

## 7. Strategic Recommendations

1. **Core Operational Focus:** While recent total profit spikes were boosted by non-operating income (`Other Income`), long-term strategy should focus on maintaining core `Sales_Growth` and defending the ~24% `OPM%` threshold.
2. **Interest Burden Management:** Financial interest expenses grew from ₹18 in FY 2015 to ₹410 in FY 2026. Monitoring debt servicing costs will protect net margins during period slowdowns.
3. **Model Enhancement:** Collect additional granular quarterly data to expand the training dataset ($N > 100$) and apply time-series cross-validation for more robust forecasting.

---

## 8. How to Run the Project

### Prerequisites
Install the required dependencies using `pip`:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### Running the Streamlit App
Execute the following command in the project directory:
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.
