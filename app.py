import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and preprocess data
@st.cache_data
def load_data():
    # Load the dataset
    df = pd.read_csv("C:/Users/HP/OneDrive/Desktop/data.csv")
    
    # Handle missing values for 'total_conversion' and 'approved_conversion'
    df['total_conversion'] = df['total_conversion'].fillna(df['total_conversion'].mean())
    df['approved_conversion'] = df['approved_conversion'].fillna(df['approved_conversion'].mean())
    
    # Convert 'gender' and 'age' to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    df['gender'] = label_encoder.fit_transform(df['gender'])  # Fit on the entire 'gender' column
    df['age'] = label_encoder.fit_transform(df['age'])
    
    # Convert date columns to datetime (necessary for 'campaign_duration')
    df['reporting_start'] = pd.to_datetime(df['reporting_start'], format='%d/%m/%Y')
    df['reporting_end'] = pd.to_datetime(df['reporting_end'], format='%d/%m/%Y')

    # Calculate 'campaign_duration' and 'engagement_rate'
    df['campaign_duration'] = (df['reporting_end'] - df['reporting_start']).dt.days
    df['engagement_rate'] = df['clicks'] / df['impressions'].replace(0, 1e-10)  # Avoid division by zero
    
    # Now that 'campaign_duration' and 'engagement_rate' are available, select features
    X_features = df[['age', 'gender', 'interest1', 'interest2', 'interest3', 'impressions', 'clicks', 'spent', 'campaign_duration', 'engagement_rate']]
    y = df['approved_conversion']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # Train the RandomForest Regressor model
    rf_model = RandomForestRegressor(random_state=42)  # Use RandomForestRegressor instead of Classifier
    rf_model.fit(X_train, y_train)
    
    return rf_model, label_encoder, df

# Load the trained Random Forest model and data
rf_model, label_encoder, df = load_data()

# Streamlit App Layout
st.title('Campaign Conversion Prediction')

# User inputs (form for input)
st.subheader("Input Campaign Details")
age = st.slider('Select Age', min_value=18, max_value=100, value=30)
gender = st.selectbox('Select Gender', options=['Male', 'Female'])  # Ensure only 'Male' and 'Female'
impressions = st.number_input('Enter Impressions', min_value=0, step=1)
clicks = st.number_input('Enter Clicks', min_value=0, step=1)
spent = st.number_input('Enter Amount Spent', min_value=0.0, step=0.01)
campaign_duration = st.number_input('Enter Campaign Duration (Days)', min_value=1, step=1)
engagement_rate = st.number_input('Enter Engagement Rate', min_value=0.0, step=0.01)

# Handle Gender encoding using the same LabelEncoder that was used during training
gender_encoded = None
if gender == 'Male':
    gender_encoded = 0  # 'Male' encoded as 0
elif gender == 'Female':
    gender_encoded = 1  # 'Female' encoded as 1

# Prepare input data for prediction if gender is encoded
if gender_encoded is not None:
    # Prepare input data for prediction
    input_data = np.array([[age, 
                            gender_encoded,  # Use the encoded gender
                            0, 0, 0,  # Placeholder for 'interest1', 'interest2', 'interest3'
                            impressions, clicks, spent, campaign_duration, engagement_rate]])

    # Standardize input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make prediction
    if st.button('Predict Conversion'):
        # Predict the likelihood of 'approved_conversion' (continuous value)
        prediction = rf_model.predict(input_data_scaled)

        # Display prediction results
        st.success(f"The predicted conversion value is: {prediction[0]:.2f}")

        # Show feature importances (insights into key features)
        feature_importances = rf_model.feature_importances_
        feature_names = ['Age', 'Gender', 'Interest1', 'Interest2', 'Interest3', 'Impressions', 'Clicks', 'Spent', 'Campaign Duration', 'Engagement Rate']
        feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        feature_df = feature_df.sort_values(by='Importance', ascending=False)

        st.subheader('Key Features Influencing the Prediction')
        st.write(feature_df)

        # Show the importance chart (bar chart)
        st.bar_chart(feature_df.set_index('Feature')['Importance'])
