import streamlit as st
import datetime
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    # Load the pre-trained model
    model = joblib.load('C:/Python/Sales Regression Model.jobli b')  # Use forward slash or escaped backslash for file path

    # Set the title of the app
    st.title("Sales Prediction with Date Inputs")

    # Get the current year for setting slider bounds
    current_year = datetime.datetime.now().year

    # Slider for the year with a defined range
    year = st.slider("Select Year", min_value=1900, max_value=current_year, value=current_year)

    # Slider for the month
    month = st.slider("Select Month", min_value=1, max_value=12, value=1)

    # Calculate the number of days in the selected month and year
    if month == 12:
        next_month = datetime.date(year + 1, 1, 1)
    else:
        next_month = datetime.date(year, month + 1, 1)

    days_in_month = (next_month - datetime.timedelta(days=1)).day

    # Slider for the day
    day = st.slider("Select Day", min_value=1, max_value=days_in_month, value=1)

    # Define the model input features
    feature_names = ['month', 'day', 'year']

    # Create a DataFrame with the correct feature names
    input_data = pd.DataFrame([[month, day, year]], columns=feature_names)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Standardize the input data
    standardized_data = scaler.fit_transform(input_data)

    # Check if Predict Sales button is pressed
    if st.button("Predict Sales"):
        # Make predictions using the standardized data
        predictions = model.predict(input_data)

        # Ensure the prediction is a float
        predicted_sales = float(predictions[0])
        st.write(f"Predicted Sales: {predicted_sales:.2f}")

    # Display the selected date
    selected_date = datetime.date(year, month, day)
    st.write("Selected Date: ", selected_date)

if __name__ == "__main__":
    main()