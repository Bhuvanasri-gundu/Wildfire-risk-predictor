from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret-key')

with open('Final_pkl_files/power_transformers_final.pkl', 'rb') as f:
    power_transformers_dict = pickle.load(f)

with open('Final_pkl_files/standard_scaler_final.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Final_pkl_files/one_hot_encoder_final.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('Final_pkl_files/Ridge_regression_V2_final.pkl', 'rb') as f:
    model = pickle.load(f)

def get_risk_level(fwi):
    if fwi <= 5:
        return "Low", "#90EE90"
    elif fwi <= 12:
        return "Moderate", "#2196F3"
    elif fwi <= 20:
        return "High", "#FFA500"
    elif fwi <= 30:
        return "Very High", "#FF6347"
    else:
        return "Extreme", "#DC143C"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    risk_level = None
    risk_color = None
    
    if request.method == 'POST':
        try:
            temp = float(request.form['temperature'])
            dew_point = float(request.form['dew_point'])
            rh = float(request.form['rh'])
            ws = float(request.form['ws'])
            pressure = float(request.form['pressure'])
            precip = float(request.form['precipitation'])
            rain_days = float(request.form['rain_days'])
            ffmc = float(request.form['ffmc'])
            dmc = float(request.form['dmc'])
            dc = float(request.form['dc'])
            bui = float(request.form['bui'])
            isi = float(request.form['isi'])
            country = request.form['country']
            
            ffmc_transformed = power_transformers_dict['fine_fuel_moisture_code'].transform([[ffmc]])[0][0]
            dmc_transformed = power_transformers_dict['duff_moisture_code'].transform([[dmc]])[0][0]
            dc_transformed = power_transformers_dict['drought_code'].transform([[dc]])[0][0]
            bui_transformed = power_transformers_dict['build_up_index'].transform([[bui]])[0][0]
            isi_transformed = power_transformers_dict['initial_spread_index'].transform([[isi]])[0][0]
            ws_transformed = power_transformers_dict['wind_speed'].transform([[ws]])[0][0]
            
            scaler_input = np.array([[
                temp, dew_point, rh, ws, pressure, precip, rain_days,
                ffmc, dmc, dc, bui, isi, 0,
                ffmc_transformed, dmc_transformed, dc_transformed, 
                bui_transformed, isi_transformed, 0, ws_transformed
            ]])
            
            sf = scaler.transform(scaler_input)[0]
            
            temp_s = sf[0]
            dew_point_s = sf[1]
            rh_s = sf[2]
            ws_s = sf[3]
            pressure_s = sf[4]
            precip_s = sf[5]
            rd_s = sf[6]
            ffmc_s = sf[7]
            dmc_s = sf[8]
            dc_s = sf[9]
            bui_s = sf[10]
            isi_s = sf[11]
            ffmc_t_s = sf[13]
            dmc_t_s = sf[14]
            dc_t_s = sf[15]
            bui_t_s = sf[16]
            isi_t_s = sf[17]
            ws_t_s = sf[19]
            
            ffmc_temp = ffmc_s * temp_s
            ffmc_rh = ffmc_s * rh_s
            rh_temp = rh_s * temp_s
            rd_ffmc = rd_s * ffmc_s
            dc_rd = dc_s * rd_s
            ws_temp = ws_s * temp_s
            ist_ws = isi_t_s * ws_t_s
            isi_ffmc = isi_s * ffmc_s
            isi_rh = isi_s * rh_s
            dmc_temp = dmc_s * temp_s
            rd_temp = rd_s * temp_s
            bui_dc = bui_s * dc_s
            bui_dmc = bui_s * dmc_s
            
            bui_sqrt = np.sqrt(bui)
            ffmc_sqrt = np.sqrt(ffmc)
            isi_sqrt = np.sqrt(isi)
            
            country_encoded = encoder.transform([[country]])
            if hasattr(country_encoded, 'toarray'):
                country_features_all = country_encoded.toarray()[0]
            else:
                country_features_all = country_encoded[0]
            
            feature_names = encoder.get_feature_names_out()
            country_dict = {name: country_features_all[i] for i, name in enumerate(feature_names)}
            
            country_canada = country_dict.get('country_Canada', 0)
            country_france = country_dict.get('country_France', 0)
            country_spain = country_dict.get('country_Spain', 0)
            country_uk = country_dict.get('country_United Kingdom', 0)
            country_us = country_dict.get('country_United States', 0)
            
            model_features = np.array([[
                temp_s, dew_point, dew_point_s, rh, rh_s, pressure, pressure_s, precip, precip_s, rain_days, rd_s,
                dmc_t_s, dc_t_s, ws_t_s,
                ffmc_temp, ffmc_rh, rh_temp, rd_ffmc, dc_rd, ws_temp, ist_ws,
                isi_ffmc, isi_rh, dmc_temp, rd_temp, bui_dc, bui_dmc,
                country_canada, country_france, country_spain, country_uk, country_us,
                bui_sqrt, ffmc_sqrt, isi_sqrt
            ]])
            
            fwi_transformed = model.predict(model_features)[0]
            
            fwi_original = power_transformers_dict['fire_weather_index'].inverse_transform([[fwi_transformed]])[0][0]
            
            prediction = round(float(fwi_original), 2)
            risk_level, risk_color = get_risk_level(prediction)
            
        except Exception as e:
            prediction = f"Error: {str(e)}"
            risk_level = "N/A"
            risk_color = "#CCCCCC"
    
    return render_template('index.html', 
                         prediction=prediction, 
                         risk_level=risk_level,
                         risk_color=risk_color)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
