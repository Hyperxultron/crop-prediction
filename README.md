# Crop Prediction System — Flask Web App

## 📁 Project Structure
```
crop_prediction/
│
├── app.py                  ← Main Flask application
├── model.pkl               ← Tumhara trained model (apna file rakho)
├── minmaxscaler.pkl        ← Tumhara scaler (apna file rakho)
├── requirements.txt        ← Dependencies
│
└── templates/
    ├── index.html          ← Home page (input form)
    └── result.html         ← Result page
```

## 🚀 Setup & Run Kaise Karein

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Apne pickle files rakho
`model.pkl` aur `minmaxscaler.pkl` files ko **app.py ke saath same folder** mein rakho.

**Important:** Agar tumhare pickle files ka naam alag hai toh `app.py` mein line 9-10 update karo:
```python
model = pickle.load(open('TUMHARA_MODEL_FILE.pkl', 'rb'))
scaler = pickle.load(open('TUMHARA_SCALER_FILE.pkl', 'rb'))
```

### Step 3: Run karo
```bash
python app.py
```

### Step 4: Browser mein open karo
```
http://127.0.0.1:5000
```

---

## ✅ Requirements
- Python 3.7+
- Flask
- scikit-learn
- numpy
