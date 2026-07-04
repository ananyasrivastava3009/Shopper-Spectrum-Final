import streamlit as st
import joblib
import pandas as pd
import os

st.title("Shopper Spectrum")
st.write("Customer Segmentation System")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "kmeans_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_online_retail.csv")

st.write(MODEL_PATH)
st.write(SCALER_PATH)

kmeans = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


df = pd.read_csv(DATA_PATH)
similarity_df = joblib.load(os.path.join(BASE_DIR, "..", "models", "similarity_df.pkl"))
def recommend_products(product_name):
    if product_name not in similarity_df.index:
        return []

    recommendations = similarity_df[product_name].sort_values(ascending=False)[1:6]
    return recommendations.index.tolist()

st.success("Models Loaded Successfully ✅")
st.header("Customer Segmentation")

recency = st.number_input("Enter Recency", min_value=0)

frequency = st.number_input("Enter Frequency", min_value=0)

monetary = st.number_input("Enter Monetary", min_value=0.0)
if st.button("Predict Cluster"):
    input_data = [[recency, frequency, monetary]]
    input_scaled = scaler.transform(input_data)
    prediction = kmeans.predict(input_scaled)

    cluster = prediction[0]

    st.success(f"Customer belongs to Cluster {cluster}")

    if cluster == 0:
        st.write("Regular Customer")
    elif cluster == 1:
        st.write("High Value Customer")
    elif cluster == 2:
        st.write("At Risk Customer")
    else:
        st.write("New Customer")

st.header("Product Recommendation")

product_name = st.text_input("Enter Product Name", key="product_name")

if st.button("Recommend Products"):
    recommendations = recommend_products(product_name)

    if len(recommendations) == 0:
        st.error("Product not found!")
    else:
        st.success("Top 5 Recommended Products")

        for product in recommendations:
            st.write(product)