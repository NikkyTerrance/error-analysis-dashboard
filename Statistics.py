"""
Distributed System Log Analytics Dashboard
Author: Graduate Data Science Application Project
Purpose: Analyze system logs to identify error patterns and predict high-risk events
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Log Analytics Dashboard", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .metric-container { background-color: #f0f2f6; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

@st.cache_data
def load_and_prepare_data(file):
    """
    Load CSV file and prepare features for analysis
    
    Returns:
        df: Prepared DataFrame with engineered features
    """
    # Load data
    df = pd.read_csv(file)
    
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Feature Engineering
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Date'] = df['Timestamp'].dt.date
    
    # Create binary target: 1 if ERROR or FATAL, 0 otherwise
    df['IsError'] = df['LogLevel'].isin(['ERROR', 'FATAL']).astype(int)
    
    # Clean TimeTaken column - remove 'ms' suffix and convert to numeric
    if df['TimeTaken'].dtype == 'object':
        df['TimeTaken'] = df['TimeTaken'].astype(str).str.replace('ms', '', regex=False).str.strip()
    
    # Convert to numeric, handling any non-numeric values
    df['TimeTaken'] = pd.to_numeric(df['TimeTaken'], errors='coerce')
    
    # Handle missing values in TimeTaken
    df['TimeTaken'] = df['TimeTaken'].fillna(df['TimeTaken'].median())
    
    return df

# ============================================================================
# EXPLORATORY DATA ANALYSIS FUNCTIONS
# ============================================================================

def display_summary_metrics(df):
    """Display key summary statistics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Events", f"{len(df):,}")
    
    with col2:
        error_rate = df['IsError'].mean() * 100
        st.metric("Error Rate", f"{error_rate:.2f}%")
    
    with col3:
        unique_services = df['Service'].nunique()
        st.metric("Unique Services", unique_services)
    
    with col4:
        unique_users = df['User'].nunique()
        st.metric("Unique Users", unique_users)
    
    with col5:
        avg_time = df['TimeTaken'].mean()
        st.metric("Avg Processing Time", f"{avg_time:.0f}ms")

def plot_events_over_time(df):
    """Visualize events over time with error highlighting - adapts to data time span"""
    
    # Calculate time span
    time_span = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 3600  # hours
    
    if time_span <= 3:  # Less than 3 hours - use minute granularity
        fig, ax = plt.subplots(figsize=(12, 5))
        
        df['DateTime_Minute'] = df['Timestamp'].dt.floor('min')
        minute_events = df.groupby('DateTime_Minute').agg({
            'IsError': ['sum', 'count']
        }).reset_index()
        minute_events.columns = ['DateTime', 'Errors', 'Total']
        minute_events = minute_events.sort_values('DateTime')
        
        if len(minute_events) > 0:
            ax.plot(minute_events['DateTime'], minute_events['Total'], 
                    label='Total Events', linewidth=1.5, color='#1f77b4', marker='o', markersize=2)
            ax.plot(minute_events['DateTime'], minute_events['Errors'], 
                    label='Error Events', linewidth=1.5, color='#d62728', marker='o', markersize=2)
            
            ax.set_xlabel('Time (Per Minute)', fontsize=12)
            ax.set_ylabel('Event Count', fontsize=12)
            ax.set_title(f'Event Timeline - Minute-by-Minute View ({time_span:.1f} hours of data)', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
        
    elif time_span <= 24:  # Less than 24 hours - use hourly granularity
        fig, ax = plt.subplots(figsize=(12, 5))
        
        df['DateTime_Hour'] = df['Timestamp'].dt.floor('H')
        hourly_events = df.groupby('DateTime_Hour').agg({
            'IsError': ['sum', 'count']
        }).reset_index()
        hourly_events.columns = ['DateTime', 'Errors', 'Total']
        hourly_events = hourly_events.sort_values('DateTime')
        
        if len(hourly_events) > 0:
            ax.plot(hourly_events['DateTime'], hourly_events['Total'], 
                    label='Total Events', linewidth=2, color='#1f77b4', marker='o', markersize=4)
            ax.plot(hourly_events['DateTime'], hourly_events['Errors'], 
                    label='Error Events', linewidth=2, color='#d62728', marker='o', markersize=4)
            
            ax.set_xlabel('Time (Hourly)', fontsize=12)
            ax.set_ylabel('Event Count', fontsize=12)
            ax.set_title(f'Event Timeline - Hourly View ({time_span:.1f} hours of data)', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
        
    else:  # More than 24 hours - use daily granularity
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Daily view
        daily_events = df.groupby('Date').agg({
            'IsError': ['sum', 'count']
        }).reset_index()
        daily_events.columns = ['Date', 'Errors', 'Total']
        daily_events['Date'] = pd.to_datetime(daily_events['Date'])
        daily_events = daily_events.sort_values('Date')
        
        if len(daily_events) > 0:
            ax1.bar(daily_events['Date'], daily_events['Total'], 
                    label='Total Events', alpha=0.7, color='#1f77b4', width=0.8)
            ax1.bar(daily_events['Date'], daily_events['Errors'], 
                    label='Error Events', alpha=0.9, color='#d62728', width=0.8)
            
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Event Count', fontsize=12)
            ax1.set_title('Daily Event Distribution', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, row in daily_events.iterrows():
                ax1.text(row['Date'], row['Total'], f"{int(row['Total']):,}", 
                        ha='center', va='bottom', fontsize=9)
        
        # Hourly view (more granular)
        df['DateTime_Hour'] = df['Timestamp'].dt.floor('H')
        hourly_events = df.groupby('DateTime_Hour').agg({
            'IsError': ['sum', 'count']
        }).reset_index()
        hourly_events.columns = ['DateTime', 'Errors', 'Total']
        hourly_events = hourly_events.sort_values('DateTime')
        
        if len(hourly_events) > 0:
            ax2.plot(hourly_events['DateTime'], hourly_events['Total'], 
                    label='Total Events', linewidth=1.5, color='#1f77b4', marker='o', markersize=3)
            ax2.plot(hourly_events['DateTime'], hourly_events['Errors'], 
                    label='Error Events', linewidth=1.5, color='#d62728', marker='o', markersize=3)
            
            ax2.set_xlabel('Date & Time (Hourly)', fontsize=12)
            ax2.set_ylabel('Event Count', fontsize=12)
            ax2.set_title('Hourly Event Timeline', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig

def plot_log_level_distribution(df):
    """Show distribution of log levels"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    log_counts = df['LogLevel'].value_counts()
    colors = {'INFO': '#2ca02c', 'WARNING': '#ff7f0e', 
              'ERROR': '#d62728', 'FATAL': '#8b0000'}
    
    bars = ax.bar(log_counts.index, log_counts.values, 
                  color=[colors.get(x, 'gray') for x in log_counts.index])
    
    ax.set_xlabel('Log Level', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Log Levels', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_time_taken_by_level(df):
    """Compare processing times across log levels"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot of TimeTaken by LogLevel
    log_levels = ['INFO', 'WARNING', 'ERROR', 'FATAL']
    log_levels = [level for level in log_levels if level in df['LogLevel'].values]
    
    data_to_plot = [df[df['LogLevel'] == level]['TimeTaken'].dropna() for level in log_levels]
    
    bp = ax1.boxplot(data_to_plot, labels=log_levels, patch_artist=True)
    
    # Color the boxes
    colors = {'INFO': '#2ca02c', 'WARNING': '#ff7f0e', 'ERROR': '#d62728', 'FATAL': '#8b0000'}
    for patch, level in zip(bp['boxes'], log_levels):
        patch.set_facecolor(colors.get(level, 'gray'))
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Log Level', fontsize=12)
    ax1.set_ylabel('Time Taken (ms)', fontsize=12)
    ax1.set_title('Processing Time Distribution by Log Level', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Average processing time by log level
    avg_time = df.groupby('LogLevel')['TimeTaken'].mean().reindex(log_levels)
    bars = ax2.bar(log_levels, avg_time.values, 
                   color=[colors.get(level, 'gray') for level in log_levels], alpha=0.7)
    
    ax2.set_xlabel('Log Level', fontsize=12)
    ax2.set_ylabel('Average Time Taken (ms)', fontsize=12)
    ax2.set_title('Average Processing Time by Log Level', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_time.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}ms', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_hourly_error_rate(df):
    """Analyze error patterns - adapts to data time span"""
    
    # Calculate time span
    time_span = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 3600  # hours
    
    if time_span <= 3:  # Less than 3 hours - use 10-minute intervals
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 10-minute interval analysis
        df['Interval_10min'] = df['Timestamp'].dt.floor('10min')
        interval_stats = df.groupby('Interval_10min').agg({
            'IsError': ['sum', 'count', 'mean']
        }).reset_index()
        interval_stats.columns = ['Time', 'Errors', 'Total', 'ErrorRate']
        interval_stats['ErrorRate'] = interval_stats['ErrorRate'] * 100
        interval_stats = interval_stats.sort_values('Time')
        
        # Error rate over time - bar chart instead of line
        bars = ax1.bar(interval_stats['Time'], interval_stats['ErrorRate'], 
                      color='#ff7f0e', alpha=0.7, width=pd.Timedelta(minutes=8))
        ax1.set_xlabel('Time (10-minute intervals)', fontsize=12)
        ax1.set_ylabel('Error Rate (%)', fontsize=12)
        ax1.set_title(f'Error Rate Over Time - 10-Minute Intervals', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, interval_stats['ErrorRate']):
            if rate > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Total events vs errors - non-overlapping bars
        x_pos = range(len(interval_stats))
        width = 0.35
        
        ax2.bar([i - width/2 for i in x_pos], interval_stats['Total'], 
               width=width, label='Total Events', alpha=0.8, color='#1f77b4')
        ax2.bar([i + width/2 for i in x_pos], interval_stats['Errors'], 
               width=width, label='Error Events', alpha=0.8, color='#d62728')
        
        ax2.set_xlabel('Time (10-minute intervals)', fontsize=12)
        ax2.set_ylabel('Event Count', fontsize=12)
        ax2.set_title('Event Volume by Time Interval', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([t.strftime('%H:%M') for t in interval_stats['Time']], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
        
    else:  # More than 3 hours - use hour of day analysis
        fig, ax = plt.subplots(figsize=(12, 5))
        
        hourly_stats = df.groupby('Hour').agg({
            'IsError': ['sum', 'count', 'mean']
        }).reset_index()
        hourly_stats.columns = ['Hour', 'Errors', 'Total', 'ErrorRate']
        hourly_stats['ErrorRate'] = hourly_stats['ErrorRate'] * 100
        
        # Create bar chart with both error rate and volume
        ax.bar(hourly_stats['Hour'], hourly_stats['ErrorRate'], 
               color='#ff7f0e', alpha=0.7, label='Error Rate (%)')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Error Rate by Hour of Day', fontsize=14, fontweight='bold')
        ax.set_xticks(hourly_stats['Hour'])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add secondary axis for total events
        ax2 = ax.twinx()
        ax2.plot(hourly_stats['Hour'], hourly_stats['Total'], 
                color='#1f77b4', marker='o', linewidth=2, markersize=6, label='Total Events')
        ax2.set_ylabel('Total Events', fontsize=12, color='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        return fig

def analyze_high_risk_entities(df):
    """Identify high-risk services and users with comparative statistics"""
    
    # Overall error rate baseline
    overall_error_rate = df['IsError'].mean() * 100
    
    # Service analysis
    service_stats = df.groupby('Service').agg({
        'IsError': ['sum', 'count', 'mean']
    }).reset_index()
    service_stats.columns = ['Service', 'Errors', 'Total', 'ErrorRate']
    service_stats['ErrorRate'] = service_stats['ErrorRate'] * 100
    service_stats['ErrorRate_vs_Avg'] = ((service_stats['ErrorRate'] - overall_error_rate) / overall_error_rate * 100)
    service_stats['ErrorRate_Diff'] = service_stats['ErrorRate'] - overall_error_rate
    service_stats = service_stats.sort_values('Errors', ascending=False).head(10)
    
    # User analysis
    user_stats = df.groupby('User').agg({
        'IsError': ['sum', 'count', 'mean']
    }).reset_index()
    user_stats.columns = ['User', 'Errors', 'Total', 'ErrorRate']
    user_stats['ErrorRate'] = user_stats['ErrorRate'] * 100
    user_stats['ErrorRate_vs_Avg'] = ((user_stats['ErrorRate'] - overall_error_rate) / overall_error_rate * 100)
    user_stats['ErrorRate_Diff'] = user_stats['ErrorRate'] - overall_error_rate
    user_stats = user_stats.sort_values('Errors', ascending=False).head(10)
    
    return service_stats, user_stats, overall_error_rate

def plot_high_risk_entities(service_stats, user_stats, overall_error_rate):
    """Visualize high-risk services and users with comparative metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top services by error count
    colors_service = ['#8b0000' if rate > overall_error_rate else '#d62728' 
                     for rate in service_stats['ErrorRate']]
    ax1.barh(service_stats['Service'], service_stats['Errors'], color=colors_service, alpha=0.7)
    ax1.set_xlabel('Error Count', fontsize=12)
    ax1.set_ylabel('Service', fontsize=12)
    ax1.set_title('Top 10 Services by Error Count', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0, color='black', linewidth=0.8)
    
    # Add comparative labels
    for i, row in service_stats.iterrows():
        label_text = f" {row['ErrorRate']:.1f}%"
        if row['ErrorRate_vs_Avg'] > 0:
            label_text += f" (+{row['ErrorRate_vs_Avg']:.0f}%)"
        ax1.text(row['Errors'], row['Service'], label_text, 
                va='center', fontsize=9, color='darkred', fontweight='bold')
    
    # Top users by error count
    colors_user = ['#cc5500' if rate > overall_error_rate else '#ff7f0e' 
                  for rate in user_stats['ErrorRate']]
    ax2.barh(user_stats['User'], user_stats['Errors'], color=colors_user, alpha=0.7)
    ax2.set_xlabel('Error Count', fontsize=12)
    ax2.set_ylabel('User', fontsize=12)
    ax2.set_title('Top 10 Users by Error Count', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='black', linewidth=0.8)
    
    # Add comparative labels
    for i, row in user_stats.iterrows():
        label_text = f" {row['ErrorRate']:.1f}%"
        if row['ErrorRate_vs_Avg'] > 0:
            label_text += f" (+{row['ErrorRate_vs_Avg']:.0f}%)"
        ax2.text(row['Errors'], row['User'], label_text, 
                va='center', fontsize=9, color='darkorange', fontweight='bold')
    
    plt.tight_layout()
    return fig

# ============================================================================
# PREDICTIVE MODELING
# ============================================================================

@st.cache_data
def build_predictive_model(df):
    """
    Build interpretable logistic regression model to predict errors
    
    Features used:
    - TimeTaken (numeric) - longer processing times may indicate issues
    - Hour (numeric) - certain hours may have more errors
    - Service frequency encoding - how often each service appears
    - User frequency encoding - activity level
    - Service error rate - historical error rate per service
    
    Returns:
        model: Trained logistic regression model
        feature_names: List of feature names
        X_test, y_test: Test data for evaluation
        training_info: Dictionary with training details
    """
    
    # Calculate error rate by service (powerful feature)
    service_error_rate = df.groupby('Service')['IsError'].mean().to_dict()
    df['Service_ErrorRate'] = df['Service'].map(service_error_rate)
    
    # Calculate error rate by user
    user_error_rate = df.groupby('User')['IsError'].mean().to_dict()
    df['User_ErrorRate'] = df['User'].map(user_error_rate)
    
    # Frequency encoding for high-cardinality features
    service_counts = df['Service'].value_counts().to_dict()
    df['Service_Frequency'] = df['Service'].map(service_counts)
    
    user_counts = df['User'].value_counts().to_dict()
    df['User_Frequency'] = df['User'].map(user_counts)
    
    # Select features for modeling
    feature_names = [
        'TimeTaken', 
        'Hour', 
        'Service_ErrorRate',
        'User_ErrorRate',
        'Service_Frequency',
        'User_Frequency'
    ]
    
    # Prepare data
    X = df[feature_names].copy()
    y = df['IsError'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train logistic regression with balanced class weights
    model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced',
        C=0.1  # Add regularization
    )
    model.fit(X_train, y_train)
    
    # Calculate training info
    training_info = {
        'total_samples': len(df),
        'error_rate': y.mean() * 100,
        'num_services': df['Service'].nunique(),
        'num_users': df['User'].nunique(),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    return model, feature_names, X_test, y_test, training_info

def display_model_performance(model, X_test, y_test, feature_names, training_info):
    """Display model performance metrics and insights"""
    
    st.subheader("📊 Model Performance")
    
    # Show training information
    st.markdown("**Training Information**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Error Rate in Data", f"{training_info['error_rate']:.2f}%")
    with col2:
        st.metric("Training Samples", f"{training_info['train_size']:,}")
    with col3:
        st.metric("Test Samples", f"{training_info['test_size']:,}")
    
    st.markdown("---")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Classification Report**")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))
    
    with col2:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    # ROC Curve
    st.markdown("**ROC Curve**")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Feature Importance (Coefficients)
    st.markdown("**Feature Importance (Logistic Regression Coefficients)**")
    st.markdown("*Positive values increase error probability; negative values decrease it*")
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title('Feature Importance in Error Prediction', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    st.title("🔍 Distributed System Log Analytics Dashboard")
    st.markdown("**Graduate Data Science Project** | Analyzing system logs to identify error patterns and predict high-risk events")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your log CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load and prepare data
        with st.spinner('Loading and preparing data...'):
            df = load_and_prepare_data(uploaded_file)
        
        st.success(f"✅ Loaded {len(df):,} log events successfully!")
        
        # Summary Metrics
        st.markdown("## 📈 Summary Metrics")
        display_summary_metrics(df)
        st.markdown("---")
        
        # Exploratory Data Analysis
        st.markdown("## 🔎 Exploratory Data Analysis")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Events Over Time", 
            "Log Level Distribution", 
            "Processing Time Analysis",
            "Temporal Patterns",
            "High-Risk Analysis"
        ])
        
        with tab1:
            st.pyplot(plot_events_over_time(df))
            time_span = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 3600
            if time_span <= 3:
                st.markdown(f"**Insight:** Your data spans {time_span:.1f} hours. The minute-by-minute view shows detailed activity patterns and identifies specific times when errors spike.")
            elif time_span <= 24:
                st.markdown(f"**Insight:** Your data spans {time_span:.1f} hours. The hourly view shows system activity trends and highlights when errors spike.")
            else:
                st.markdown("**Insight:** This chart shows daily and hourly system activity trends, highlighting when errors spike.")
        
        with tab2:
            st.pyplot(plot_log_level_distribution(df))
            st.markdown("**Insight:** Most events are informational, but ERROR and FATAL events require attention.")
        
        with tab3:
            st.pyplot(plot_time_taken_by_level(df))
            st.markdown("**Insight:** Compare processing times across different log levels to identify performance issues.")
        
        with tab4:
            st.pyplot(plot_hourly_error_rate(df))
            time_span = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 3600
            if time_span <= 3:
                st.markdown(f"**Insight:** With {time_span:.1f} hours of data, this shows error patterns in 10-minute intervals. Identifies specific time windows with elevated error rates and traffic spikes.")
            else:
                st.markdown("**Insight:** Certain hours may have higher error rates, indicating load or maintenance windows. The dual-axis view shows both error rate and total traffic volume.")
        
        with tab5:
            service_stats, user_stats, overall_error_rate = analyze_high_risk_entities(df)
            st.pyplot(plot_high_risk_entities(service_stats, user_stats, overall_error_rate))
            
            st.markdown(f"""
            **Insight:** Identifies which services and users are associated with the most errors. 
            
            **Baseline:** Overall system error rate is **{overall_error_rate:.2f}%**
            
            - **Services:** Bars show total error count. Darker red = error rate above average. The percentage labels show error rate and comparative difference (e.g., "+150%" means 150% higher than average).
            - **Users:** Similar analysis for users. High error counts might indicate automated systems, power users encountering edge cases, or potentially malicious activity.
            
            Services/users **above average** require immediate investigation. Those **below average** despite high error counts may just be high-volume entities.
            """)
            
            # Show detailed tables
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Service Details**")
                display_cols = ['Service', 'Errors', 'Total', 'ErrorRate', 'ErrorRate_vs_Avg']
                service_display = service_stats[display_cols].copy()
                service_display.columns = ['Service', 'Errors', 'Total Events', 'Error Rate (%)', '% vs Average']
                service_display['% vs Average'] = service_display['% vs Average'].apply(lambda x: f"+{x:.0f}%" if x > 0 else f"{x:.0f}%")
                st.dataframe(service_display.style.background_gradient(subset=['Error Rate (%)'], cmap='Reds'))
            with col2:
                st.markdown("**User Details**")
                display_cols = ['User', 'Errors', 'Total', 'ErrorRate', 'ErrorRate_vs_Avg']
                user_display = user_stats[display_cols].copy()
                user_display.columns = ['User', 'Errors', 'Total Events', 'Error Rate (%)', '% vs Average']
                user_display['% vs Average'] = user_display['% vs Average'].apply(lambda x: f"+{x:.0f}%" if x > 0 else f"{x:.0f}%")
                st.dataframe(user_display.style.background_gradient(subset=['Error Rate (%)'], cmap='Oranges'))
        
        st.markdown("---")
        
        # Predictive Modeling
        st.markdown("## 🤖 Predictive Modeling: Error Event Classification")
        st.markdown("""
        **Objective:** Predict whether a log event is an ERROR or FATAL using logistic regression.
        
        **Why Logistic Regression?**
        - Interpretable: Each coefficient shows how a feature impacts error probability
        - Fast and efficient for deployment
        - Provides probability scores for risk assessment
        - Industry standard for binary classification in production systems
        """)
        
        # Add diagnostic analysis before modeling
        st.markdown("### 🔍 Pre-Model Diagnostics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Check if errors are random across services
            service_error_variance = df.groupby('Service')['IsError'].mean().std()
            st.metric("Service Error Rate Variance", f"{service_error_variance:.4f}")
            if service_error_variance < 0.05:
                st.caption("⚠️ Low variance - errors appear random across services")
        
        with col2:
            # Check correlation between TimeTaken and errors
            time_corr = df[['TimeTaken', 'IsError']].corr().iloc[0, 1]
            st.metric("TimeTaken-Error Correlation", f"{time_corr:.4f}")
            if abs(time_corr) < 0.1:
                st.caption("⚠️ Weak correlation - processing time doesn't predict errors")
        
        with col3:
            # Check if errors cluster by hour
            hour_error_variance = df.groupby('Hour')['IsError'].mean().std()
            st.metric("Hourly Error Rate Variance", f"{hour_error_variance:.4f}")
            if hour_error_variance < 0.02:
                st.caption("⚠️ Low variance - errors not time-dependent")
        
        st.markdown("---")
        
        with st.spinner('Training predictive model...'):
            model, feature_names, X_test, y_test, training_info = build_predictive_model(df)
        
        display_model_performance(model, X_test, y_test, feature_names, training_info)
        
        st.markdown("---")
        
        # Key Insights Section
        st.markdown("## 💡 Key Insights & Recommendations")
        
        # Calculate some insights
        error_rate = df['IsError'].mean() * 100
        
        # Get detailed service and user statistics
        service_stats_full, user_stats_full, overall_error_rate = analyze_high_risk_entities(df)
        
        top_error_service = service_stats_full.iloc[0]['Service'] if len(service_stats_full) > 0 else "N/A"
        top_error_user = user_stats_full.iloc[0]['User'] if len(user_stats_full) > 0 else "N/A"
        peak_error_hour = df.groupby('Hour')['IsError'].mean().idxmax()
        
        # Get top error counts and comparative stats
        if len(service_stats_full) > 0:
            top_service_errors = service_stats_full.iloc[0]['Errors']
            top_service_error_rate = service_stats_full.iloc[0]['ErrorRate']
            top_service_vs_avg = service_stats_full.iloc[0]['ErrorRate_vs_Avg']
        else:
            top_service_errors = 0
            top_service_error_rate = 0
            top_service_vs_avg = 0
            
        if len(user_stats_full) > 0:
            top_user_errors = user_stats_full.iloc[0]['Errors']
            top_user_error_rate = user_stats_full.iloc[0]['ErrorRate']
            top_user_vs_avg = user_stats_full.iloc[0]['ErrorRate_vs_Avg']
        else:
            top_user_errors = 0
            top_user_error_rate = 0
            top_user_vs_avg = 0
        
        # Check AUC to determine predictability
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        if auc < 0.6:
            model_insight = f"""
        4. **Model Performance (AUC = {auc:.3f}):** The predictive model shows near-random performance, indicating that **errors in this system are not strongly predictable** from the available features (service, user, time, processing duration). This suggests:
           - Errors may be triggered by external factors not captured in the logs (network issues, database problems, third-party API failures)
           - Error generation may be truly stochastic/random in nature
           - Additional features would be needed (e.g., system metrics, resource utilization, request payload characteristics)
           - **This is a valid finding**: Not all phenomena are predictable, and recognizing this is important for data scientists
        """
        else:
            model_insight = f"""
        4. **Predictive Model (AUC = {auc:.3f}):** The logistic regression model achieves moderate predictive performance and can be deployed to flag high-risk events in real-time.
        """
        
        insights = f"""
        1. **Overall System Health:** The system has an error rate of {error_rate:.2f}%, with {df['IsError'].sum():,} error events out of {len(df):,} total events.
        
        2. **High-Risk Service - '{top_error_service}':**
           - **Error Count:** {int(top_service_errors):,} errors
           - **Error Rate:** {top_service_error_rate:.2f}% (system average: {overall_error_rate:.2f}%)
           - **Comparison:** This service has **{abs(top_service_vs_avg):.0f}% {'higher' if top_service_vs_avg > 0 else 'lower'}** error rate than the system average
           - **Priority:** {'🔴 CRITICAL - Immediate investigation required' if top_service_vs_avg > 100 else '🟡 ELEVATED - Monitor closely' if top_service_vs_avg > 0 else '🟢 NORMAL - High volume but acceptable rate'}
        
        3. **High-Risk User - '{top_error_user}':**
           - **Error Count:** {int(top_user_errors):,} errors
           - **Error Rate:** {top_user_error_rate:.2f}% (system average: {overall_error_rate:.2f}%)
           - **Comparison:** This user has **{abs(top_user_vs_avg):.0f}% {'higher' if top_user_vs_avg > 0 else 'lower'}** error rate than the system average
           - **Potential Causes:**
             - {'🤖 Likely an automated system/bot with configuration issues' if top_user_vs_avg > 100 else '👤 Power user encountering edge cases' if top_user_vs_avg > 0 else '✅ High-volume user with normal error rate'}
             - {'⚠️ Consider security review for potential malicious activity' if top_user_vs_avg > 200 else ''}
        
        4. **Temporal Patterns:** Error rates peak around hour {peak_error_hour}, suggesting potential load or maintenance issues during this time.
        
        {model_insight}
        
        6. **Actionable Next Steps:**
           - **If errors are predictable (AUC > 0.6):** Deploy model for real-time alerting
           - **If errors are random (AUC < 0.6):** Focus on root cause analysis rather than prediction
           - **Service Investigation:** Deep-dive into '{top_error_service}' logs - it's performing {'significantly worse' if top_service_vs_avg > 100 else 'worse'} than average
           - **User Behavior Analysis:** Review '{top_error_user}' activity patterns - {'immediate action needed' if top_user_vs_avg > 200 else 'monitoring recommended' if top_user_vs_avg > 0 else 'appears normal despite high volume'}
           - **Comparative Approach:** Focus resources on entities with error rates **above average**, not just high error counts
           - Investigate error message patterns using NLP (TF-IDF, keyword extraction)
           - Collect additional system metrics (CPU, memory, network latency)
           - Implement comprehensive logging to capture more context around failures
        """
        
        st.markdown(insights)
        
        # Data Preview
        with st.expander("📄 View Raw Data Sample"):
            st.dataframe(df.head(100))
    
    else:
        st.info("👆 Please upload a CSV file containing distributed system logs to begin analysis.")
        st.markdown("""
        **Expected CSV format:**
        - `Timestamp` – Event timestamp
        - `LogLevel` – Severity (INFO, WARNING, ERROR, FATAL)
        - `Service` – Service identifier
        - `Message` – Event description
        - `RequestID` – Unique request ID
        - `User` – Associated user
        - `ClientIP` – Client IP address
        - `TimeTaken` – Processing time in milliseconds
        """)

if __name__ == "__main__":
    main()