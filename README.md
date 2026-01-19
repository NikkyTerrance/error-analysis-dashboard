# Distributed System Log Analytics Dashboard

> A graduate-level data science project demonstrating analytical skills, statistical thinking, and effective data communication through interactive visualization.

## 📋 Project Overview

This project analyzes distributed system logs to identify patterns in system behavior, predict error events, and provide actionable insights for system reliability and security monitoring. Built as part of a graduate data science and analytics application, it emphasizes interpretability, clear analysis, and practical recommendations over complex machine learning.

**Author:** Graduate Data Science Application  
**Focus Area:** Cybersecurity & System Analytics  
**Key Technologies:** Python, Streamlit, Scikit-learn, Pandas, Matplotlib

---

## 🎯 Project Objectives

1. **Exploratory Data Analysis** - Understand patterns in system activity and log severity
2. **Error Pattern Detection** - Identify factors associated with ERROR and FATAL events
3. **Predictive Modeling** - Build interpretable models to assess error predictability
4. **Interactive Visualization** - Present findings through an accessible dashboard
5. **Actionable Insights** - Provide clear recommendations for system improvement

---

## 🗂️ Dataset Structure

The project works with CSV files containing distributed system log data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | DateTime | Date and time of each event (e.g., `2023-11-20T08:40:50.664842`) |
| `LogLevel` | String | Severity of the event (`INFO`, `WARNING`, `ERROR`, `FATAL`) |
| `Service` | String | Name or identifier of the service generating the log |
| `Message` | String | Text description of the event |
| `RequestID` | Integer | Unique identifier for each request |
| `User` | String | User associated with the event |
| `ClientIP` | String | Client or application IP address |
| `TimeTaken` | String | Time taken to process the request (e.g., `28ms`, `55ms`) |

**Expected Data:** 10,000+ rows of realistic system and security log data

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Required Packages

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

Place your CSV file (e.g., `logdata.csv`) in the project directory or have it ready to upload through the dashboard.

### Step 3: Run the Application

```bash
streamlit run python.py
```

Or using Python module syntax:

```bash
python -m streamlit run python.py
```

The dashboard will automatically open in your default browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### 1. Summary Metrics
- Total event count
- Overall error rate
- Number of unique services
- Average processing time

### 2. Exploratory Data Analysis

#### Events Over Time
- **Adaptive granularity:** Automatically adjusts based on data time span
  - < 3 hours: Minute-by-minute view
  - 3-24 hours: Hourly view
  - > 24 hours: Daily + hourly views
- Highlights error spikes and traffic patterns

#### Log Level Distribution
- Visual breakdown of INFO, WARNING, ERROR, and FATAL events
- Color-coded for easy interpretation

#### Processing Time Analysis
- Box plots showing distribution by log level
- Average processing time comparison
- Identifies performance issues by severity

#### Temporal Patterns
- Error rate analysis by time interval
- Identifies high-risk time windows
- Event volume correlation with error rates

#### High-Risk Analysis
- **Top 10 services by error count** with comparative metrics
- **Top 10 users by error count** with comparative metrics
- **Comparative analysis:** Shows percentage above/below system average
- **Color coding:** Darker colors indicate above-average error rates
- Detailed tables with exportable data

### 3. Predictive Modeling

#### Model Approach: Logistic Regression
**Why this model?**
- Highly interpretable - coefficients show exact impact
- Industry standard for binary classification
- Provides probability scores for risk assessment
- Fast enough for real-time deployment

#### Features Used
1. **Service_ErrorRate** - Historical error rate for each service
2. **User_ErrorRate** - Historical error rate for each user
3. **Service_Frequency** - How often each service is called
4. **User_Frequency** - User activity level
5. **TimeTaken** - Processing duration (milliseconds)
6. **Hour** - Time of day (0-23)

#### Pre-Model Diagnostics
The dashboard includes diagnostic checks to assess predictability:
- Service error rate variance
- TimeTaken-Error correlation
- Hourly error rate variance

These diagnostics explain *why* a model performs well or poorly.

#### Performance Metrics
- Classification report (precision, recall, F1-score)
- Confusion matrix
- ROC curve with AUC score
- Feature importance (coefficient analysis)

### 4. Key Insights & Recommendations

Automatically generated insights include:
- Overall system health assessment
- High-risk service identification with comparative analysis
- High-risk user identification with security context
- Temporal pattern analysis
- Model performance interpretation
- Actionable next steps based on findings

---

## 🧠 Key Technical Decisions

### 1. Why Logistic Regression Over Deep Learning?
- **Interpretability:** Stakeholders need to understand *why* predictions are made
- **Defensibility:** Can explain every coefficient in an interview or audit
- **Appropriate complexity:** Matches the problem scope and data characteristics
- **Production ready:** Simple enough to deploy and maintain

### 2. Handling Class Imbalance
- Used `class_weight='balanced'` to prevent model bias toward majority class
- This is critical when ERROR/FATAL events are rare (often 1-5% of logs)

### 3. Feature Engineering Strategy
- **Avoided label encoding** for high-cardinality categoricals (Service, User)
- Instead used **frequency encoding** and **target encoding** (error rates)
- This preserves meaningful relationships without arbitrary numeric assignments

### 4. Adaptive Visualizations
- Charts automatically adjust granularity based on data time span
- Ensures insights are visible regardless of dataset duration
- Demonstrates thoughtful consideration of data characteristics

### 5. When Models Don't Work
- If AUC < 0.6, the dashboard explains this is a valid finding
- Provides context: errors may be random or require additional features
- Recommends appropriate next steps (root cause analysis vs. prediction)

---

## 📈 Sample Insights

### When the Model Works Well (AUC > 0.7)
*"The logistic regression model achieved an AUC of 0.78, indicating that errors are moderately predictable from the available features. The Service_ErrorRate feature has the strongest coefficient (2.34), showing that services with historical errors are 10x more likely to generate future errors. This model can be deployed for real-time alerting."*

### When the Model Doesn't Work (AUC < 0.6)
*"The model achieved an AUC of 0.52, essentially random performance. Pre-model diagnostics revealed low variance in error rates across services (0.015) and weak correlation between processing time and errors (r=0.03). This indicates errors are likely triggered by external factors not captured in the logs, such as network issues or database problems. Recommendation: Focus on root cause analysis and collect additional telemetry."*

---

## 🎓 Interview Talking Points

### Analytical Approach
*"I started with exploratory analysis to understand the data before jumping to modeling. The visualizations revealed that only 2% of events were errors, which informed my decision to use class-balanced logistic regression."*

### Model Selection
*"I chose logistic regression over random forests or neural networks because in a security operations center, analysts need to understand why an alert was triggered. The coefficients show exactly which factors increase risk, making the model trustworthy and auditable."*

### Handling Negative Results
*"The low AUC score is actually a valuable finding. It tells us that errors aren't systematically caused by specific services or users - they're essentially random given our features. This guided my recommendations toward collecting additional metrics rather than deploying an ineffective model."*

### Comparative Analysis
*"Rather than just showing which services have the most errors, I implemented comparative analysis. A service with 1,000 errors might be performing well if it handles 100,000 requests (1% error rate), while a service with 100 errors could be critical if it only handles 200 requests (50% error rate). The comparative metrics reveal the true outliers."*

### Cybersecurity Relevance
*"Log analysis is fundamental to security monitoring. This approach extends directly to intrusion detection, anomaly identification, and threat hunting. The same principles apply whether you're predicting system errors or detecting malicious activity."*

---

## 📁 Project Structure

```
project/
│
├── python.py              # Main Streamlit dashboard application
├── logdata.csv           # Sample log data (not included in repo)
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── screenshots/          # Dashboard screenshots (optional)
```

---

## 🔧 Requirements.txt

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## 🐛 Troubleshooting

### Issue: "streamlit: command not found"
**Solution:** Use `python -m streamlit run python.py` instead

### Issue: "TypeError: Cannot convert ['28ms' '55ms'...] to numeric"
**Solution:** The code automatically strips 'ms' suffix and converts to numeric. Ensure you're using the latest version of the script.

### Issue: Empty charts or "no data"
**Solution:** Verify your CSV has the correct column names and data types. Check the console for specific error messages.

### Issue: AUC score is 0.5 (random)
**Solution:** This is often a valid finding, not an error. The dashboard explains this and provides appropriate recommendations.

---

## 🎯 Future Enhancements

### Short Term
- [ ] NLP analysis on error messages (TF-IDF, keyword extraction)
- [ ] Anomaly detection using Isolation Forest
- [ ] Export functionality for reports and visualizations
- [ ] Configurable alerting thresholds

### Long Term
- [ ] Time series forecasting for error prediction
- [ ] Integration with real-time log streaming (e.g., Kafka)
- [ ] Multi-model comparison (Logistic Regression vs. Random Forest)
- [ ] User authentication and role-based access

---

## 📚 Learning Outcomes

This project demonstrates:

✅ **Data Preparation:** Loading, cleaning, and feature engineering  
✅ **Exploratory Analysis:** Statistical visualization and pattern recognition  
✅ **Statistical Modeling:** Logistic regression with proper evaluation  
✅ **Critical Thinking:** Recognizing when models don't work and why  
✅ **Communication:** Translating technical findings into actionable insights  
✅ **Software Engineering:** Building production-ready interactive applications  
✅ **Domain Knowledge:** Understanding log analysis in cybersecurity context  

---

## 📞 Contact & Feedback

This project was created as part of a graduate data science application. For questions or feedback, please reach out via:

- **Email:** [your.email@example.com]
- **LinkedIn:** [Your LinkedIn Profile]
- **GitHub:** [Your GitHub Profile]

---

## 📄 License

This project is created for educational and application purposes. Feel free to use as a reference for your own graduate applications or learning projects.

---

## 🙏 Acknowledgments

- **Anthropic Claude** - For assistance in project development and documentation
- **Streamlit** - For providing an excellent framework for data applications
- **Scikit-learn** - For robust machine learning tools

---

**Last Updated:** January 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✅
