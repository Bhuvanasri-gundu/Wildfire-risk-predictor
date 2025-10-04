# Tempest: FWI Predictor

## Project Overview
The **Fire Weather Index (FWI) Predictor** is a machine learning project designed to forecast wildfire risk based on real-time environmental data. Wildfires pose significant threats to ecosystems and communities, and FWI is a crucial metric for proactive wildfire management. This project leverages **Ridge Regression** to predict FWI and provides a **Flask-based web application** for live predictions.

## Features
- Predict FWI using environmental inputs such as Temperature, Humidity, Wind Speed, Rain, FFMC, DMC, ISI, and Region.
- Data preprocessing including missing value handling, outlier detection, and feature scaling with **StandardScaler**.
- Web-based interface built with Flask to allow users to input values and receive predictions.
- Model evaluation with **MAE, RMSE, and R² metrics** to ensure prediction reliability.

## Project Modules
1. **Data Collection & Preprocessing**  
   - Collected and cleaned dataset, handled missing values, encoded categorical features.  
   - Explored data distributions and feature relationships using visualizations.

2. **Feature Engineering & Model Training**  
   - Selected key features, scaled numeric data, and split dataset into train/test sets.  
   - Trained **Ridge Regression model** to predict FWI and saved the model (`ridge.pkl`) and scaler (`scaler.pkl`).

3. **Evaluation & Optimization**  
   - Evaluated model performance using MAE, RMSE, and R².  
   - Tuned alpha parameter to optimize predictions.

4. **Deployment via Flask**  
   - Built a **Flask web app** with input form and prediction output.  
   - Integrated model and scaler for real-time inference.


