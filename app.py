import streamlit as st
import psycopg2 as pg
import joblib
from datetime import datetime

# Load model (assuming the model is in the same directory)
model = joblib.load("Random_Forest_model.joblib")

# Function to connect to the database and fetch data
def fetch_data(query):
    try:
        connection = pg.connect(
            dbname="enabl_crm",
            user="power_bi_user",
            password="with_power_comes_great_bi_001",
            host="192.168.250.32",
            port="5444"
        )
        cursor = connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except (Exception, pg.Error) as error:
        st.error(f"Error while connecting to PostgreSQL: {error}")
        return []
    finally:
        if connection:
            cursor.close()
            connection.close()

# Feature selection with constraints and input handling
def main():
    st.title("Random Forest Model Predictor")

    # Dropdown for probability
    probability = st.selectbox('Probability', ['', 40, 70, 90, 100])

    # Date input fields
    today = datetime.now().date()

    quote_sent_date = st.date_input('Quote Sent Date', value=today)
    order_lost_date = st.date_input('Order Lost Date', value=today)
    end_date = st.date_input('End Date', value=today)
    start_date = st.date_input('Start Date', value=today)
    request_deadline = st.date_input('Request Deadline', value=today)

    # Validate dates to ensure quote_sent_date and order_lost_date are not in the future
    current_datetime = datetime.now()
    if quote_sent_date > today:
        st.error("Quote Sent Date cannot be in the future.")
    if order_lost_date > today:
        st.error("Order Lost Date cannot be in the future.")

    # Continue only if dates are valid
    if quote_sent_date <= today and order_lost_date <= today:
        # Convert selected dates to seconds relative to the current date and time
        quote_sent_seconds = (datetime.combine(quote_sent_date, datetime.min.time()) - current_datetime).total_seconds()
        order_lost_seconds = (datetime.combine(order_lost_date, datetime.min.time()) - current_datetime).total_seconds()
        end_seconds = (datetime.combine(end_date, datetime.min.time()) - current_datetime).total_seconds()
        start_seconds = (datetime.combine(start_date, datetime.min.time()) - current_datetime).total_seconds()
        request_deadline_seconds = (datetime.combine(request_deadline, datetime.min.time()) - current_datetime).total_seconds()

        # Dropdowns and other inputs
        tender_status_options = ['', "Open", "Search", "Evaluation", "Ready for Quote", "Failed/Rejected", "Solution"]
        tender_status = st.selectbox("Tender Status ID", tender_status_options)

        company_code_options = [
            '', "Enabl DK", "Enabl India", "Enabl China", "Enabl UK", "ENABL US",
            "EWS Taiwan Branch", "HEJ Engineering UA", "Sonne Bulgaria EOOD", "ENABL HU"
        ]
        company_code = st.selectbox("Company Code ID", company_code_options)

        case_status_options = ['', "Active", "Mistake", "Lost", "Closed"]
        case_status = st.selectbox("Case Status ID", case_status_options)

        case_lost_reason_options = [
            '', "Not prioritized", "No capacity available", "Too expensive", "Too late",
            "Cancelled by customer", "Lost to competitor", "Lost because of management decision"
        ]
        case_lost_reason = st.selectbox("Case Lost Reason ID", case_lost_reason_options)

        quote_status_options = ['', "Send to customer", "Negotiation", "Decision pending", "Failed/Rejected"]
        quote_status = st.selectbox("Quote Status ID", quote_status_options)

        value_weighted_dkk = st.number_input('Value Weighted DKK', min_value=0, max_value=1_000_000_000)
        value_dkk = st.number_input('Value DKK', min_value=0, max_value=1_000_000_000)
        forecast_weighted_dkk = st.number_input('Forecast Weighted DKK', min_value=0, max_value=1_000_000_000)

        contact_list = fetch_data("SELECT name, id FROM contacts")
        contact_options = [""] + [name for name, _ in contact_list]
        contact = st.selectbox("Contact ID", contact_options)
        contact_id = dict(contact_list).get(contact, 0)

        tender_manager_list = fetch_data(
            "SELECT CONCAT(e.first_name, ' ', e.last_name) AS name, e.id "
            "FROM cases c JOIN employees e ON c.tender_manager_id = e.id"
        )
        tender_manager_options = [""] + [name for name, _ in tender_manager_list]
        tender_manager = st.selectbox("Tender Manager ID", tender_manager_options)
        tender_manager_id = dict(tender_manager_list).get(tender_manager, 0)

        project_type_list = fetch_data("SELECT title, id FROM project_types")
        project_type_options = [""] + [title for title, _ in project_type_list]
        project_type = st.selectbox("Project Type ID", project_type_options)
        project_type_id = dict(project_type_list).get(project_type, 0)

        hour_approval_list = fetch_data(
            "SELECT CONCAT(e.first_name, ' ', e.last_name) AS name, e.id "
            "FROM cases c JOIN employees e ON c.hour_approval_responsible_id = e.id"
        )
        hour_approval_options = [""] + [name for name, _ in hour_approval_list]
        hour_approval_responsible = st.selectbox("Hour Approval Responsible ID", hour_approval_options)
        hour_approval_responsible_id = dict(hour_approval_list).get(hour_approval_responsible, 0)

        nav_project_number = st.number_input('NAV Project Number', min_value=0)

        # Example conditions for any possible binary features
        nav_project_number_nan = 0 if nav_project_number else 1  # Assuming this indicates absence
        nav_project_number_P = 0  # Placeholder for another indicator related to nav_project_number

        features = [
            probability, quote_sent_seconds, nav_project_number_nan, order_lost_seconds, 
            nav_project_number_P, 1 if tender_status == '' else tender_status_options.index(tender_status), 
            1 if company_code == '' else company_code_options.index(company_code), 
            1 if case_status == '' else case_status_options.index(case_status), end_seconds, start_seconds,
            1 if case_lost_reason == '' else case_lost_reason_options.index(case_lost_reason), 
            1 if quote_status == '' else quote_status_options.index(quote_status), request_deadline_seconds,
            value_weighted_dkk, value_dkk, contact_id, tender_manager_id,
            forecast_weighted_dkk, project_type_id, hour_approval_responsible_id
        ]

        # Define a mapping for the predicted case phase IDs to meaningful class names
        prediction_mapping = {
            1: "Lead",
            2: "Quote",
            3: "Agreement",
            4: "Tender",
            5: "Prospect"
        }

        if st.button('Predict'):
            # Get probabilities for each class
            probabilities = model.predict_proba([features])[0]

            # Get the predicted class index
            predicted_index = model.predict([features])[0]

            # Get the class name using the mapping
            predicted_class = prediction_mapping.get(predicted_index, "Unknown")

            st.write(f'Predicted Case Phase: {predicted_class}')
            st.write('Probabilities for each class:')
            for i, prob in enumerate(probabilities, start=1):
                class_name = prediction_mapping.get(i, f"Class {i}")
                st.write(f'{class_name}: {prob:.2f}')

if __name__ == "__main__":
    main()
