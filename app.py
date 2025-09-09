import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import holidays
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import torch
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data if not already present
# Use st.cache_resource to avoid re-downloading on each run
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
         nltk.download('vader_lexicon')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()


# Load the trained model, preprocessor, and feature names
# Use st.cache_resource to load these only once
@st.cache_resource
def load_resources():
    try:
        with open('tuned_xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('pca_transformer.pkl', 'rb') as f:
            pca_transformer = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        return model, preprocessor, feature_names, pca_transformer, tfidf_vectorizer

    except FileNotFoundError:
        st.error("Error: Model, preprocessor, feature names, PCA, or TF-IDF files not found.")
        st.stop()


model, preprocessor, feature_names, pca_transformer, tfidf_vectorizer = load_resources()


# Initialize tokenizer and model for embeddings
@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

tokenizer, embedding_model = load_embedding_model()

def get_embedding(text):
    text = str(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# Function to engineer features (including text and graph features)
def engineer_features(df, feature_names):
    st.write("--- Engineering Features ---")
    st.write("Initial DataFrame columns:", df.columns.tolist())

    # Ensure date columns are in datetime format
    for date_col in ['Incident_Date', 'Claim_Submission_Date', 'Policy_Start_Date', 'Policy_End_Date']:
         if date_col in df.columns:
            # Use errors='coerce' to turn unparseable dates into NaT
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)


    # --- Temporal Features ---
    # Initialize temporal features to default values if they don't exist
    df['Incident_on_Holiday'] = 0
    df['Incident_on_Weekend'] = 0
    df['Claim_Submission_on_Weekend'] = 0
    df['Days_to_Claim_Submission'] = -1 # Default to -1 for missing
    df['Late_Claim_Submission'] = 0
    df['Policy_Duration_Days'] = -1 # Default to -1 for missing
    df['Claim_Within_2Months_of_Start'] = 0
    df['Claim_Within_2Months_of_End'] = 0


    if 'Incident_Date' in df.columns:
        # Incident on Holiday
        try:
            start_year = df['Incident_Date'].min().year if df['Incident_Date'].min() is not pd.NaT else 2023 # Default year if no valid dates
            end_year = df['Incident_Date'].max().year if df['Incident_Date'].max() is not pd.NaT else 2025 # Default year if no valid dates
            years = range(start_year, end_year + 1)
            nigerian_holidays = holidays.Nigeria(years=years)
            df['Incident_on_Holiday'] = df['Incident_Date'].apply(lambda date: 1 if pd.notna(date) and date in nigerian_holidays else 0)
        except Exception as e:
            st.warning(f"Could not fully create 'Incident_on_Holiday': {e}")

        # Incident on Weekend
        df['Incident_on_Weekend'] = df['Incident_Date'].dt.dayofweek.apply(lambda x: 1 if pd.notna(x) and x >= 5 else 0)

    if 'Claim_Submission_Date' in df.columns:
        # Claim Submission on Weekend
        df['Claim_Submission_on_Weekend'] = df['Claim_Submission_Date'].dt.dayofweek.apply(lambda x: 1 if pd.notna(x) and x >= 5 else 0)

    if 'Claim_Submission_Date' in df.columns and 'Incident_Date' in df.columns:
        # Days to Claim Submission
        df['Days_to_Claim_Submission'] = (df['Claim_Submission_Date'] - df['Incident_Date']).dt.days.fillna(-1) # Use -1 for missing

        # Late Claim Submission (>= 90 days)
        df['Late_Claim_Submission'] = (df['Days_to_Claim_Submission'] >= 90).astype(int)


    if 'Policy_Start_Date' in df.columns and 'Policy_End_Date' in df.columns:
        # Policy Duration
        df['Policy_Duration_Days'] = (df['Policy_End_Date'] - df['Policy_Start_Date']).dt.days.fillna(-1) # Use -1 for missing

    # Claims within 2 months of policy start/end dates
    if 'Claim_Submission_Date' in df.columns and 'Policy_Start_Date' in df.columns:
         df['Claim_Within_2Months_of_Start'] = ((df['Claim_Submission_Date'] - df['Policy_Start_Date']).dt.days <= 60).astype(int)

    if 'Claim_Submission_Date' in df.columns and 'Policy_End_Date' in df.columns:
        df['Claim_Within_2Months_of_End'] = ( ((df['Policy_End_Date'] - df['Claim_Submission_Date']).dt.days <= 60) &
                                            ((df['Policy_End_Date'] - df['Claim_Submission_Date']).dt.days >= 0)).astype(int)


    # --- Claim Frequency Features ---
    # Initialize frequency features to default values if they don't exist
    df['Claim_Count_2Years'] = 0
    df['Frequent_Claimant'] = 0
    df['Customer_Claim_Count'] = 0
    df['Frequent_Customer_Claimant'] = 0
    df['Prior_Fraudulent_Claim'] = 0 # Default to 0 for a new customer or if no prior data

    # For single row input, frequency based on that row is 1.
    # For batch processing (CSV upload), these should be calculated based on the batch or historical data.
    # For simplicity here, we'll assume single input or rely on merging with historical data if available.
    # If processing a CSV, you would need to implement logic here to calculate these based on the CSV data.
    # For now, keeping defaults for new claims/customers.


    # --- Claim Amount Features ---
    # Initialize claim amount features to default values if they don't exist
    df['High_Claim_Amount_Flag'] = 0
    df['Claim_vs_Premium_Ratio'] = 0.0

    if 'Claim_Amount' in df.columns:
        # High Claim amount Flag (using a predefined threshold from training)
        # This threshold (percentile_90) should be saved from the training phase
        # For this example, let's use a placeholder value. Replace with your actual saved value.
        percentile_90_threshold = 454548.349 # REPLACE WITH YOUR SAVED 90TH PERCENTILE
        df['High_Claim_Amount_Flag'] = (df['Claim_Amount'] > percentile_90_threshold).astype(int)


    if 'Claim_Amount' in df.columns and 'Premium_Amount' in df.columns:
        # Calculate the ratio of Claim_Amount to Premium_Amount
        df['Claim_vs_Premium_Ratio'] = df.apply(lambda row: row['Claim_Amount'] / row['Premium_Amount'] if pd.notna(row['Claim_Amount']) and pd.notna(row['Premium_Amount']) and row['Premium_Amount'] > 0 else 0, axis=1)


    # --- Text Features (TF-IDF and Embeddings) ---
    # Initialize text features to default values if they don't exist
    if 'Adjuster_Notes' not in df.columns:
        df['Adjuster_Notes'] = '' # Initialize with empty string if missing

    # TF-IDF features (using fitted vectorizer)
    try:
        # Ensure all TF-IDF columns from training are initialized
        for col in tfidf_vectorizer.get_feature_names_out():
            if col not in df.columns:
                df[col] = 0.0

        tfidf_matrix = tfidf_vectorizer.transform(df['Adjuster_Notes'].fillna('')) # Handle potential NaN in Notes
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=df.index)
        # Drop existing TF-IDF columns before concatenating to avoid duplicates
        cols_to_drop = [col for col in tfidf_df.columns if col in df.columns]
        if cols_to_drop:
             df = df.drop(columns=cols_to_drop)
        df = pd.concat([df, tfidf_df], axis=1)

    except Exception as e:
         st.warning(f"Could not fully create TF-IDF features: {e}")


    # Embedding features (using loaded model and PCA transformer)
    try:
        # Ensure all Embedding/PCA columns from training are initialized
        for i in range(pca_transformer.n_components):
            if f'embed_pca_{i+1}' not in df.columns:
                df[f'embed_pca_{i+1}'] = 0.0

        # Handle potential errors during embedding calculation for a row
        def safe_get_embedding(text):
            try:
                return get_embedding(text)
            except Exception as e:
                st.warning(f"Error getting embedding for text: {text[:50]}... Error: {e}")
                return np.zeros(embedding_model.config.hidden_size) # Return zeros array on error

        df['adjuster_embedding'] = df['Adjuster_Notes'].apply(safe_get_embedding)

        # Ensure embedding matrix has consistent shape even with errors
        embedding_list = df['adjuster_embedding'].tolist()
        # Check if any embeddings are non-empty before stacking
        if all(isinstance(emb, np.ndarray) and emb.size > 0 for emb in embedding_list):
             embedding_matrix = np.vstack(embedding_list)
             reduced_embeddings = pca_transformer.transform(embedding_matrix) # Use transform
             embedding_pca_df = pd.DataFrame(reduced_embeddings, columns=[f'embed_pca_{i+1}' for i in range(pca_transformer.n_components)], index=df.index)
             # Drop existing embed_pca columns before concatenating
             cols_to_drop = [col for col in embedding_pca_df.columns if col in df.columns]
             if cols_to_drop:
                 df = df.drop(columns=cols_to_drop)
             df = pd.concat([df.drop(columns=['adjuster_embedding']), embedding_pca_df], axis=1)
        else:
            st.warning("No valid embeddings generated. Skipping PCA transformation and adding placeholder columns.")
            # Add placeholder columns if no valid embeddings
            for i in range(pca_transformer.n_components):
                 df[f'embed_pca_{i+1}'] = 0.0
            if 'adjuster_embedding' in df.columns:
                 df = df.drop(columns=['adjuster_embedding'])


    except Exception as e:
         st.warning(f"Could not fully create Embedding/PCA features: {e}")


    # --- Graph-based Features (Centrality) ---
    # Initialize centrality features to default values if they don't exist
    df['Customer_Centrality'] = 0.0
    df['Location_Centrality'] = 0.0
    # For single row input or small batches, centrality cannot be calculated accurately.
    # Relying on defaults or merging with pre-calculated values from training data.


    # --- Sentiment Analysis ---
    # Initialize sentiment features to default values if they don't exist
    df['Sentiment_Score'] = 0.0
    df['Negative_Tone_Flag'] = 0

    if 'Adjuster_Notes' in df.columns:
        sia = SentimentIntensityAnalyzer()
        df['Sentiment_Score'] = df['Adjuster_Notes'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
        df['Negative_Tone_Flag'] = (df['Sentiment_Score'] < -0.5).astype(int)


    st.write("DataFrame columns after engineering features:", df.columns.tolist())
    st.write("Shape after engineering:", df.shape)

    # Ensure all expected feature_names columns are present after engineering
    for col in feature_names:
        if col not in df.columns:
            st.warning(f"Engineered feature '{col}' is missing. Adding with default value.")
            # Determine a default value based on expected column type or a reasonable placeholder
            # This requires knowing the expected dtypes of your final features.
            # For simplicity, using 0.0 for numerical/binary and 'missing' for categorical.
            # A more robust approach would involve saving the dtypes during training.
            if col.startswith('tfidf_') or col.startswith('embed_pca_') or '_Flag' in col or '_on_' in col or '_Days' in col or '_Count' in col or '_Ratio' in col or 'Centrality' in col or 'Sentiment_Score' in col or (col in df.columns and df[col].dtype in [np.number, np.int64, np.float64]): # Assuming numerical/binary
                 df[col] = 0.0
            else: # Assuming categorical (will be handled by one-hot encoder's imputer)
                 df[col] = 'missing' # Or a specific placeholder used in training


    return df


# Function to preprocess data
def preprocess_data(df, preprocessor, feature_names):
    st.write("--- Preprocessing Data ---")
    st.write("DataFrame columns before final preprocessing:", df.columns.tolist())
    st.write("Shape before final preprocessing:", df.shape)


    # Ensure the order of columns matches the training data and only keep necessary columns
    # This also handles missing columns by dropping those not in feature_names
    # and relies on the preprocessor's imputer for handling NaNs in the selected columns.
    # First, add any missing columns that are in feature_names but not in the current df
    # This is a safeguard if engineer_features somehow missed initializing a column.
    for col in feature_names:
        if col not in df.columns:
             st.warning(f"Column '{col}' not found before final preprocessing. Adding with default.")
             # Determine a default value based on expected column type or a reasonable placeholder
             # This requires knowing the expected dtypes of your final features.
             # For simplicity, using 0.0 for numerical/binary and 'missing' for categorical.
             if col.startswith('tfidf_') or col.startswith('embed_pca_') or '_Flag' in col or '_on_' in col or '_Days' in col or '_Count' in col or '_Ratio' in col or 'Centrality' in col or 'Sentiment_Score' in col or (col in df.columns and df[col].dtype in [np.number, np.int64, np.float64]): # Assuming numerical/binary
                  df[col] = 0.0
             else: # Assuming categorical (will be handled by one-hot encoder's imputer)
                  df[col] = 'missing' # Or a specific placeholder used in training


    # Now, align and select columns
    try:
        df_aligned = df[feature_names]
        st.write("DataFrame columns after aligning order:", df_aligned.columns.tolist())
        st.write("Shape after aligning:", df_aligned.shape)

        # Apply the fitted preprocessor
        # The preprocessor's SimpleImputer will handle NaNs in numerical features
        # The OneHotEncoder's handle_unknown='ignore' and SimpleImputer will handle new/missing categories
        processed_data = preprocessor.transform(df_aligned)
        st.write("Preprocessing successful.")
        st.write("Shape after preprocessing:", processed_data.shape)
        return processed_data

    except ValueError as ve:
        st.error(f"An error occurred during data preprocessing: {ve}")
        st.error("This often happens if the columns in the input data do not match the expected features.")
        st.write("Expected features:", feature_names)
        st.write("Columns in DataFrame before preprocessing:", df.columns.tolist())
        st.stop()
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during data preprocessing: {e}")
        st.stop()
        return None


# Streamlit App Title
st.title("Smart Claims Fraud Detection System")

st.markdown("""
This application predicts whether an insurance claim is fraudulent based on the provided details.
You can either input claim details manually or upload a CSV file containing multiple claims.
""")

# Option to upload CSV or enter manually
option = st.radio("Select input method:", ("Manual Input", "Upload CSV"))

if option == "Manual Input":
    st.header("Enter Claim Details")

    # Input fields for claim details (manual)
    col1, col2 = st.columns(2)

    with col1:
        # Example: Numerical Inputs
        claim_amount = st.number_input("Claim Amount", min_value=0.0, value=10000.0)
        premium_amount = st.number_input("Premium Amount", min_value=0.0, value=5000.0)
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
        # Add other numerical inputs as needed

    with col2:
        # Example: Categorical Inputs
        location = st.selectbox("Location", ['Ibadan', 'Port Harcourt', 'Abuja', 'Kano', 'Lagos']) # Replace with your actual locations
        policy_type = st.selectbox("Policy Type", ['Family', 'Corporate', 'Individual']) # Replace with your actual policy types
        claim_type = st.selectbox("Claim Type", ['Health', 'Life', 'Auto', 'Fire', 'Gadget']) # Replace with your actual claim types
        incident_type = st.selectbox("Incident Type", ['Fire', 'Death', 'Accident', 'Theft', 'Illness']) # Replace with your actual incident types
        customer_gender = st.selectbox("Customer Gender", ['Female', 'Male']) # Replace with your actual genders
        customer_occupation = st.selectbox("Customer Occupation", ['Artisan', 'Unemployed', 'Student', 'Teacher', 'Engineer', 'Trader', 'Driver']) # Replace with your actual occupations
        # Add other categorical inputs as needed

    # Example: Date Inputs
    incident_date = st.date_input("Incident Date")
    claim_submission_date = st.date_input("Claim Submission Date")
    policy_start_date = st.date_input("Policy Start Date")
    policy_end_date = st.date_input("Policy End Date")

    # Example: Text Input
    adjuster_notes = st.text_area("Adjuster Notes", "Enter notes here...")

    # Create a dictionary from the input values
    input_data = {
        'Claim_Amount': claim_amount,
        'Premium_Amount': premium_amount,
        'Customer_Age': customer_age,
        'Location': location,
        'Policy_Type': policy_type,
        'Claim_Type': claim_type,
        'Incident_Type': incident_type,
        'Customer_Gender': customer_gender,
        'Customer_Occupation': customer_occupation,
        'Incident_Date': incident_date,
        'Claim_Submission_Date': claim_submission_date,
        'Policy_Start_Date': policy_start_date,
        'Policy_End_Date': policy_end_date,
        'Adjuster_Notes': adjuster_notes,
        # Add other necessary columns with placeholder/default values
        'Prior_Fraudulent_Claim': 0, # Default for a new claim
        'Claim_Count_2Years': 0, # Default for a new claim
        'Customer_Claim_Count': 0, # Default for a new claim
        'Policy_Number': 'NEW_POLICY', # Placeholder
        'Customer_Name': 'NEW_CUSTOMER', # Placeholder
        'Customer_Email': 'new@example.com', # Placeholder
        'Customer_Phone': '123-456-7890', # Placeholder
        'Claim_ID': 'NEW_CLAIM_ID' # Placeholder

    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    st.write("--- Input DataFrame ---")
    st.write(input_df)


    # Add a button to make predictions
    if st.button("Predict Fraud"):
        # Engineer features
        engineered_df = engineer_features(input_df.copy(), feature_names)

        # Preprocess data
        processed_input = preprocess_data(engineered_df.copy(), preprocessor, feature_names)

        if processed_input is not None:
            # Make prediction
            prediction = model.predict(processed_input)
            prediction_proba = model.predict_proba(processed_input)[:, 1] if hasattr(model, 'predict_proba') else None

            st.header("Prediction Result")
            if prediction[0] == 1:
                st.error("Prediction: Fraudulent Claim")
            else:
                st.success("Prediction: Non-Fraudulent Claim")

            if prediction_proba is not None:
                st.write(f"Probability of Fraud: {prediction_proba[0]:.4f}")

            # Optional: SHAP explanation for this prediction (for single manual input)
            st.header("Explanation (SHAP Values)")
            try:
                single_explainer = shap.TreeExplainer(model)
                single_shap_values = single_explainer.shap_values(processed_input)
                shap.initjs()
                st.write(shap.force_plot(single_explainer.expected_value, single_shap_values[0,:], processed_input[0,:]))

            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {e}")


elif option == "Upload CSV":
    st.header("Upload Claim Data (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file
            csv_df = pd.read_csv(uploaded_file)
            st.write("--- Uploaded Data ---")
            st.write(csv_df.head())

            # Add placeholder/default columns that might be missing in the uploaded CSV
            required_placeholder_cols = ['Policy_Number', 'Customer_Name', 'Customer_Email', 'Customer_Phone', 'Claim_ID']
            for col in required_placeholder_cols:
                 if col not in csv_df.columns:
                      csv_df[col] = f'MISSING_{col}' # Add with a placeholder

            # Engineer features for the uploaded data
            engineered_csv_df = engineer_features(csv_df.copy(), feature_names)

            # Preprocess the engineered data
            processed_csv_data = preprocess_data(engineered_csv_df.copy(), preprocessor, feature_names)

            if processed_csv_data is not None:
                # Make predictions on the uploaded data
                predictions = model.predict(processed_csv_data)
                predictions_proba = model.predict_proba(processed_csv_data)[:, 1] if hasattr(model, 'predict_proba') else None

                # Add predictions to the original DataFrame
                csv_df['Predicted_Fraud_Flag'] = predictions
                if predictions_proba is not None:
                    csv_df['Fraud_Probability'] = predictions_proba

                st.header("Prediction Summary")

                # Display prediction counts
                fraud_count = csv_df['Predicted_Fraud_Flag'].sum()
                total_claims = len(csv_df)
                legitimate_count = total_claims - fraud_count

                st.write(f"Total Claims Processed: {total_claims}")
                st.write(f"Predicted Fraudulent Claims: {fraud_count}")
                st.write(f"Predicted Legitimate Claims: {legitimate_count}")

                # Plot a pie chart of the results
                st.header("Prediction Distribution")
                labels = ['Fraudulent', 'Legitimate']
                sizes = [fraud_count, legitimate_count]
                colors = ['#ff9999','#66b3in1'] # Red for Fraudulent, Green for Legitimate
                explode = (0.1, 0)  # explode 1st slice

                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                        shadow=True, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)

                # Display the DataFrame with predictions
                st.header("Detailed Predictions")
                st.write(csv_df[['Claim_ID', 'Predicted_Fraud_Flag', 'Fraud_Probability']].head()) # Display relevant columns
                if total_claims > 5:
                    st.write(f"... showing first 5 rows out of {total_claims}")

                # Option to download results
                csv_output = csv_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_output,
                    file_name='claim_predictions.csv',
                    mime='text/csv',
                )


        except Exception as e:
            st.error(f"Error processing the uploaded CSV file: {e}")
            st.error("Please ensure the CSV file has the expected columns and data format.")
