"""
Streamlit App for Customer Churn Prediction
Interactive dashboard for making predictions.
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData
from src.logger import logger


# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'prediction_pipeline' not in st.session_state:
    try:
        with st.spinner("Loading prediction model..."):
            st.session_state.prediction_pipeline = PredictionPipeline()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


def main():
    """Main function for Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict customer churn using machine learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    st.sidebar.info("""
    **About This App**
    
    This dashboard predicts customer churn using machine learning.
    
    **Features:**
    - Single customer prediction
    - Batch prediction from CSV
    - Interactive visualizations
    - Risk assessment
    """)
    
    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Model Information")
    model_type = type(st.session_state.prediction_pipeline.model).__name__
    st.sidebar.write(f"**Model Type:** {model_type}")
    st.sidebar.write(f"**Status:** 🟢 Active")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction", "📈 Analytics"])
    
    # Tab 1: Single Prediction
    with tab1:
        single_prediction_page()
    
    # Tab 2: Batch Prediction
    with tab2:
        batch_prediction_page()
    
    # Tab 3: Analytics
    with tab3:
        analytics_page()


def single_prediction_page():
    """Page for single customer prediction."""
    st.header("🎯 Single Customer Prediction")
    st.write("Enter customer information below to predict churn probability.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Basic Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("📱 Phone Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        
        st.subheader("💳 Billing Information")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col2:
        st.subheader("🌐 Internet Services")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        st.subheader("💰 Charges")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=0.1)
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
        # Prepare data
        customer_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Make prediction
        with st.spinner("Analyzing customer data..."):
            try:
                result = st.session_state.prediction_pipeline.predict(customer_data)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Create columns for results
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    churn_pred = result['prediction']
                    color = "red" if churn_pred == "Yes" else "green"
                    st.markdown(f"### Prediction: **:{color}[{churn_pred}]**")
                
                with res_col2:
                    st.metric("Churn Probability", f"{result['churn_probability']*100:.2f}%")
                
                with res_col3:
                    risk_level = result['risk_level']
                    risk_color = {
                        'Low': 'green', 'Medium': 'orange', 
                        'High': 'red', 'Critical': 'darkred'
                    }.get(risk_level, 'gray')
                    st.markdown(f"### Risk Level: **:{risk_color}[{risk_level}]**")
                
                # Probability gauge
                st.markdown("---")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['churn_probability'] * 100,
                    title={'text': "Churn Probability (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 50], 'color': "yellow"},
                            {'range': [50, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("---")
                st.subheader("💡 Recommendations")
                if result['risk_level'] in ['High', 'Critical']:
                    st.error("""
                    **High Churn Risk Detected!**
                    - Immediate intervention recommended
                    - Contact customer with retention offer
                    - Review account for service issues
                    - Consider loyalty rewards or discounts
                    """)
                elif result['risk_level'] == 'Medium':
                    st.warning("""
                    **Medium Churn Risk**
                    - Monitor customer engagement
                    - Send satisfaction survey
                    - Highlight service benefits
                    """)
                else:
                    st.success("""
                    **Low Churn Risk**
                    - Customer appears satisfied
                    - Continue standard engagement
                    - Consider upsell opportunities
                    """)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")


def batch_prediction_page():
    """Page for batch predictions."""
    st.header("📊 Batch Prediction")
    st.write("Upload a CSV file with customer data for batch predictions.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {len(df)} customers")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            # Predict button
            if st.button("🔮 Predict for All Customers", type="primary"):
                with st.spinner(f"Making predictions for {len(df)} customers..."):
                    try:
                        # Make predictions
                        results_df = st.session_state.prediction_pipeline.predict_from_dataframe(df)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Customers", len(results_df))
                        
                        with col2:
                            churn_count = (results_df['prediction'] == 'Yes').sum()
                            st.metric("Predicted Churners", churn_count)
                        
                        with col3:
                            churn_rate = (churn_count / len(results_df)) * 100
                            st.metric("Churn Rate", f"{churn_rate:.1f}%")
                        
                        with col4:
                            high_risk = (results_df['risk_level'].isin(['High', 'Critical'])).sum()
                            st.metric("High Risk", high_risk)
                        
                        # Risk distribution
                        st.markdown("---")
                        st.subheader("📈 Risk Distribution")
                        
                        risk_counts = results_df['risk_level'].value_counts()
                        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                    title="Customer Risk Distribution",
                                    color_discrete_sequence=px.colors.sequential.RdYlGn_r)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.markdown("---")
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Show sample format
        st.subheader("📄 Expected CSV Format")
        sample_df = pd.DataFrame({
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'tenure': [12, 24],
            'MonthlyCharges': [70.0, 85.0],
            '...': ['...', '...']
        })
        st.dataframe(sample_df)


def analytics_page():
    """Analytics and insights page."""
    st.header("📈 Analytics & Insights")
    
    st.info("This section will show model performance metrics and feature importance.")
    
    # Feature importance
    try:
        importance = st.session_state.prediction_pipeline.get_feature_importance()
        
        if importance:
            st.subheader("🎯 Feature Importance")
            
            # Convert to dataframe
            importance_df = pd.DataFrame({
                'Feature': list(importance.keys())[:15],
                'Importance': list(importance.values())[:15]
            })
            
            # Plot
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', title="Top 15 Most Important Features")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available for this model type")
    
    except Exception as e:
        st.error(f"Error loading feature importance: {str(e)}")


if __name__ == "__main__":
    main()