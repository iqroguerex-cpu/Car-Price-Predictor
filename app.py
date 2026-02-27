import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------
dataset = pd.read_csv("Car_Price.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# -----------------------------
# Handle Missing Values
# -----------------------------
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:5])
X[:, 1:5] = imputer.transform(X[:, 1:5])

# -----------------------------
# Encode Brand
# -----------------------------
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# -----------------------------
# Feature Scaling
# -----------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# =============================
# STREAMLIT UI
# =============================

st.title("🚗 Car Price Prediction App")
st.markdown("Predict car prices using Linear Regression.")

st.sidebar.header("Enter Car Details")

brand = st.sidebar.selectbox("Brand", ["Toyota", "Honda", "BMW"])
engine = st.sidebar.slider("Engine Size (L)", 1.0, 4.0, 2.0)
horsepower = st.sidebar.slider("Horsepower", 80, 350, 150)
mileage = st.sidebar.slider("Mileage", 10000, 80000, 40000)
age = st.sidebar.slider("Car Age (Years)", 0, 10, 3)

input_df = pd.DataFrame(
    [[brand, engine, horsepower, mileage, age]],
    columns=["Brand", "EngineSize", "Horsepower", "Mileage", "CarAge"]
)

# Apply preprocessing
input_X = ct.transform(input_df)
input_X = sc.transform(input_X)

if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_X)
    st.success(f"💰 Estimated Car Price: ${prediction[0]:,.0f}")

# =============================
# VISUALIZATIONS
# =============================

st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

# 1️⃣ Price Distribution
with col1:
    fig1 = plt.figure()
    plt.hist(dataset["Price"], bins=10)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    st.pyplot(fig1)

# 2️⃣ Mileage vs Price
with col2:
    fig2 = plt.figure()
    sns.scatterplot(x="Mileage", y="Price", data=dataset)
    plt.title("Mileage vs Price")
    st.pyplot(fig2)

# 3️⃣ Brand vs Price
st.subheader("🏷 Brand Impact on Price")
fig3 = plt.figure()
sns.boxplot(x="Brand", y="Price", data=dataset)
plt.title("Brand vs Price")
st.pyplot(fig3)

# =============================
# MODEL PERFORMANCE
# =============================

st.subheader("📈 Model Performance")

st.write(f"R² Score: **{r2:.2f}**")
st.write(f"Mean Absolute Error: **${mae:,.0f}**")

fig4 = plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
st.pyplot(fig4)
