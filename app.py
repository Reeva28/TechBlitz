import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_processor import load_data, get_basic_stats, prepare_data
from model import train_xgboost_model, predict_with_model, get_feature_importance
from visualization import (
    plot_correlation_heatmap, 
    plot_feature_distribution, 
    plot_feature_importance,
    plot_scatter_matrix, 
    plot_boxplots_by_target,
    plot_actual_vs_predicted
)

# Set page configuration
st.set_page_config(
    page_title="Air Quality Analysis & Prediction",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("🌍 Air Quality Analysis & Prediction Dashboard")
st.markdown("""
This dashboard provides interactive visualizations and predictive analysis of air quality data using XGBoost.
""")

# Load data
@st.cache_data
def get_data():
    return load_data('attached_assets/cleaned_data.csv')

df = get_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Correlation Analysis", "Feature Analysis", "Predictive Modeling"])

# Data Overview Page
if page == "Data Overview":
    st.header("Data Overview")
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("Basic Statistics")
    stats = get_basic_stats(df)
    st.dataframe(stats)
    
    # Target distribution
    st.subheader("Target Variable Distribution")
    
    target_col = st.selectbox(
        "Select Target Variable", 
        ["Air Quality_Hazardous", "Air Quality_Moderate", "Air Quality_Poor"]
    )
    
    # Count of True/False values
    target_counts = df[target_col].value_counts().reset_index()
    target_counts.columns = [target_col, 'Count']
    
    # Create two columns for pie chart and bar chart
    col1, col2 = st.columns(2)
