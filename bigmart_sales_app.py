# bigmart_sales_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# App title
st.title("🛒 BigMart Sales Prediction App")

# Sidebar
st.sidebar.title("Upload Datasets")
train_file = st.sidebar.file_uploader("Upload Train.csv", type=["csv"])
test_file = st.sidebar.file_uploader("Upload  Test.csv", type=["csv"])
 
# Function to preprocess data
def preprocess_data(train, test):
    train['source'] = 'train'
    test['source'] = 'test'
    data = pd.concat([train, test], ignore_index=True)

    data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
    data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)
    data['Item_Visibility'] = data['Item_Visibility'].replace(0, data['Item_Visibility'].mean())

    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
        'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
    })

    data['Outlet_Years'] = 2025 - data['Outlet_Establishment_Year']

    le = LabelEncoder()
    cols = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type']
    for col in cols:
        data[col] = le.fit_transform(data[col])

    data = pd.get_dummies(data, columns=['Item_Type', 'Outlet_Identifier'])

    data.drop(['Item_Identifier', 'Outlet_Establishment_Year', 'source'], axis=1, inplace=True)

    train = data[:train.shape[0]]
    test = data[train.shape[0]:]
    return train, test

# When both files are uploaded
if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    test_original = test_df.copy()

    st.subheader("Raw Training Data")
    st.dataframe(train_df.head())

    # Preprocess
    train_processed, test_processed = preprocess_data(train_df, test_df)

    X = train_processed.drop(['Item_Outlet_Sales'], axis=1)
    y = train_processed['Item_Outlet_Sales']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        st.success(f"✅ Model trained! Validation RMSE: {rmse:.2f}")

        # Plot actual vs predicted
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_val, y=y_pred, ax=ax)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        st.pyplot(fig)

        # Predictions on test data
        test_processed.drop(['Item_Outlet_Sales'], axis=1, inplace=True, errors='ignore')
        predictions = model.predict(test_processed)

        submission = pd.DataFrame({
            "Item_Identifier": test_original["Item_Identifier"],
            "Outlet_Identifier": test_original["Outlet_Identifier"],
            "Item_Outlet_Sales": predictions
        })

        csv = submission.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Submission CSV",
            data=csv,
            file_name='bigmart_submission.csv',
            mime='text/csv'
        )
 
