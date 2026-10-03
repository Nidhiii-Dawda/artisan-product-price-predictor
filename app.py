
import streamlit as st
import pandas as pd
import joblib

# Load model
model_data = joblib.load("artisan_price_model.pkl")
model = model_data["model"]
features = model_data["features"]

st.title("Artisan Product Price Predictor")
st.write("Enter product details to predict the price.")

craft_type = st.selectbox("Craft Type", [
    "Pottery", "Wood Carving", "Handloom", "Embroidery",
    "Bamboo", "Terracotta", "Metal Craft", "Stone Craft",
    "Block Printing", "Folk Painting", "Jewellery", "Paper Craft"
])

material = st.selectbox("Material", [
    "Clay", "Wood", "Cotton", "Bamboo", "Metal",
    "Stone", "Silk", "Paper", "Leather"
])

region = st.selectbox("Region", [
    "Maharashtra", "Rajasthan", "Gujarat", "Assam",
    "West Bengal", "Odisha", "Karnataka", "Tamil Nadu"
])

product_size = st.selectbox("Product Size", ["Small", "Medium", "Large"])

production_time = st.number_input("Production Time (Hours)", min_value=1.0)
material_cost = st.number_input("Material Cost", min_value=0.0)
labor_cost = st.number_input("Labor Cost", min_value=0.0)
shipping_cost = st.number_input("Shipping Cost", min_value=0.0)
previous_price = st.number_input("Previous Price", min_value=0.0)
previous_sales = st.number_input("Previous Sales", min_value=0.0)
units_produced = st.number_input("Units Produced", min_value=1.0)
rating = st.number_input("Customer Rating", min_value=0.0, max_value=5.0)
season = st.selectbox("Season", ["Normal", "Festival", "Winter", "Summer", "Monsoon"])
festival = st.selectbox("Festival", ["None", "Diwali", "Holi", "Christmas", "Eid", "Dussehra"])
online_views = st.number_input("Online Views", min_value=0.0)
discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0)
competition = st.selectbox("Competition Level", ["Low", "Medium", "High"])
marketing_spend = st.number_input("Marketing Spend", min_value=0.0)
customer_inquiries = st.number_input("Customer Inquiries", min_value=0.0)
experience = st.number_input("Seller Experience (Years)", min_value=0.0)
month = st.number_input("Month", min_value=1, max_value=12)
sales_channel = st.selectbox("Sales Channel", ["Online", "Offline", "Both"])

if st.button("Predict Price"):

    total_cost = material_cost + labor_cost
    cost_per_hour = total_cost / production_time

    data = pd.DataFrame([{
        "Craft_Type": craft_type,
        "Material": material,
        "Region": region,
        "Product_Size": product_size,
        "Production_Time_Hours": production_time,
        "Material_Cost": material_cost,
        "Labor_Cost": labor_cost,
        "Shipping_Cost": shipping_cost,
        "Previous_Price": previous_price,
        "Previous_Sales": previous_sales,
        "Units_Produced": units_produced,
        "Customer_Rating": rating,
        "Season": season,
        "Festival": festival,
        "Online_Views": online_views,
        "Discount_Percent": discount,
        "Competition_Level": competition,
        "Marketing_Spend": marketing_spend,
        "Customer_Inquiries": customer_inquiries,
        "Seller_Experience_Years": experience,
        "Month": month,
        "Sales_Channel": sales_channel,
        "Total_Cost": total_cost,
        "Cost_Per_Hour": cost_per_hour
    }])

    data = pd.get_dummies(data, drop_first=True)
    data = data.reindex(columns=features, fill_value=0)

    prediction = model.predict(data)[0]

    st.success(f"Predicted Price: ₹{prediction:,.2f}")
