import joblib
import numpy as np

# Load model dan scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Contoh input [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
sample = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
sample_scaled = scaler.transform(sample)

# Prediksi
prediction = model.predict(sample_scaled)
print("Hasil prediksi:", "Penyakit Jantung" if prediction[0]==1 else "Sehat")

