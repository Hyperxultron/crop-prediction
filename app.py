from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import requests

app = Flask(__name__)

UNSPLASH_ACCESS_KEY = "HCHIedHPZ4aIQfDy55-Oehx9ZJYiSt_WjJh03QgCLto"

try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    scaler = None

def get_crop_image(crop_name):
    try:
        url = f"https://api.unsplash.com/search/photos?query={crop_name} crop farm&per_page=1&client_id={UNSPLASH_ACCESS_KEY}"
        response = requests.get(url)
        data = response.json()
        if data['results']:
            return data['results'][0]['urls']['regular']
    except:
        pass
    return None

def get_crop_info(crop_name):
    crop_info = {
        'rice': {'season': 'Kharif', 'region': 'Punjab, Haryana, UP', 'info': 'Paddy crop requiring high water and humidity.'},
        'maize': {'season': 'Kharif/Rabi', 'region': 'Karnataka, AP, Rajasthan', 'info': 'Versatile crop used for food and fodder.'},
        'jute': {'season': 'Kharif', 'region': 'West Bengal, Bihar', 'info': 'Natural fiber crop grown in humid conditions.'},
        'cotton': {'season': 'Kharif', 'region': 'Gujarat, Maharashtra, Punjab', 'info': 'Major cash crop requiring black soil.'},
        'coconut': {'season': 'Year-round', 'region': 'Kerala, Tamil Nadu, Karnataka', 'info': 'Tropical crop requiring coastal climate.'},
        'papaya': {'season': 'Year-round', 'region': 'Andhra Pradesh, Gujarat', 'info': 'Fast growing fruit crop.'},
        'orange': {'season': 'Winter', 'region': 'Nagpur, Rajasthan', 'info': 'Citrus fruit requiring cool winters.'},
        'apple': {'season': 'Summer', 'region': 'Himachal Pradesh, Kashmir', 'info': 'Temperate fruit requiring cold climate.'},
        'muskmelon': {'season': 'Summer', 'region': 'UP, Punjab, Rajasthan', 'info': 'Summer fruit requiring warm dry climate.'},
        'watermelon': {'season': 'Summer', 'region': 'Karnataka, Tamil Nadu', 'info': 'Warm season crop requiring sandy soil.'},
        'grapes': {'season': 'Winter', 'region': 'Maharashtra, Karnataka', 'info': 'Vine fruit requiring well-drained soil.'},
        'mango': {'season': 'Summer', 'region': 'UP, Andhra Pradesh, Maharashtra', 'info': 'King of fruits requiring tropical climate.'},
        'banana': {'season': 'Year-round', 'region': 'Tamil Nadu, Maharashtra, Gujarat', 'info': 'Tropical fruit requiring humid conditions.'},
        'pomegranate': {'season': 'Year-round', 'region': 'Maharashtra, Gujarat, Rajasthan', 'info': 'Drought resistant fruit crop.'},
        'lentil': {'season': 'Rabi', 'region': 'MP, UP, Rajasthan', 'info': 'Pulse crop requiring cool dry climate.'},
        'blackgram': {'season': 'Kharif', 'region': 'AP, Tamil Nadu, UP', 'info': 'Pulse crop rich in protein.'},
        'mungbean': {'season': 'Kharif', 'region': 'Rajasthan, Maharashtra, AP', 'info': 'Short duration pulse crop.'},
        'mothbeans': {'season': 'Kharif', 'region': 'Rajasthan, Gujarat', 'info': 'Drought tolerant pulse crop.'},
        'pigeonpeas': {'season': 'Kharif', 'region': 'Maharashtra, Karnataka, MP', 'info': 'Long duration pulse crop.'},
        'kidneybeans': {'season': 'Kharif', 'region': 'J&K, Himachal Pradesh, UP', 'info': 'Nutritious pulse crop.'},
        'chickpea': {'season': 'Rabi', 'region': 'MP, Rajasthan, Maharashtra', 'info': 'Most important pulse crop of India.'},
        'coffee': {'season': 'Year-round', 'region': 'Karnataka, Kerala, Tamil Nadu', 'info': 'Plantation crop requiring hilly terrain.'},
    }
    return crop_info.get(crop_name.lower(), {
        'season': 'Varies', 'region': 'India', 'info': 'A recommended crop for your soil conditions.'
    })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = round(max(probabilities) * 100, 1)

        crop_name = str(prediction).capitalize()
        crop_image = get_crop_image(crop_name)
        crop_info = get_crop_info(crop_name)

        return render_template('result.html',
                             crop=crop_name,
                             confidence=confidence,
                             crop_image=crop_image,
                             crop_info=crop_info)
    except Exception as e:
        return render_template('result.html', crop=None, error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)