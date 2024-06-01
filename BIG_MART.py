import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = 'BIG_MART.pkl'  # Replace with your model's path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define a function for prediction
def preprocess_input(data):
    # Perform label encoding for categorical features
    encoder = LabelEncoder()
    for col in ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','Item_Identifier','Outlet_Identifier']:
        data[col] = encoder.fit_transform(data[col])
    return data

def predict_sales(data):
    data = preprocess_input(pd.DataFrame(data, index=[0]))
    prediction = model.predict(data)
    return prediction[0]

# Streamlit app
st.title("Big Mart Sales Prediction")

st.header("Enter the details of the item and outlet")

item_identifier = st.text_input("Item Identifier")
item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
item_type = st.selectbox("Item Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"])
outlet_identifier = st.selectbox("Outlet Identifier", ["OUT049", "OUT018", "OUT010", "OUT013", "OUT027", "OUT045", "OUT017", "OUT046", "OUT035", "OUT019"])
outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2","Supermarket Type3","Grocery Store"])
item_mrp =  st.number_input("Item_MRP", min_value=0.0, max_value=10000.0, key="float_input")
Outlet_Establishment_Year= st.number_input("Outlet Establishment Year",min_value=1990,max_value=2010)
Item_Weight=st.number_input("Item Weight",min_value=0,max_value=50)
Item_Visibility=st.number_input("Item Visibility",min_value=0,max_value=1)
# Create a dictionary with the inputs
data = {
    'Item_Identifier': item_identifier,
    'Item_Weight': Item_Weight,
    'Item_Fat_Content': item_fat_content,
    'Item_Visibility': Item_Visibility,
    'Item_Type': item_type,
    'Item_MRP': item_mrp,
    'Outlet_Identifier': outlet_identifier,
    'Outlet_Establishment_Year': Outlet_Establishment_Year,
    'Outlet_Size': outlet_size,
    'Outlet_Location_Type': outlet_location_type,
    'Outlet_Type': outlet_type

}

# Predict button
if st.button("Predict Sales"):
    sales = predict_sales(data)
    st.write(f"Predicted Item Outlet Sales: ${sales:.2f}")



