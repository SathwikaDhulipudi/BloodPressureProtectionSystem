# ==========================================
# Blood Pressure / Hypertension Prediction
# With Console Input
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv(r"C:\Users\konda\Downloads\archive\hypertension_data.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]


# ==========================================
# 3. Handle Missing Values
# ==========================================

imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# ==========================================
# 4. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# 5. Train Random Forest Model
# ==========================================

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print("\nModel Training Completed!")
print("Accuracy:", accuracy_score(y_test, rf.predict(X_test)))


# ==========================================
# 6. Console Input for New Patient
# ==========================================

print("\n===== Enter Patient Details =====")

age = int(input("Enter Age: "))
sex = int(input("Enter Sex (1=Male, 0=Female): "))
cp = int(input("Chest Pain Type (0-3): "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol Level: "))
fbs = int(input("Fasting Blood Sugar > 120 (1=True, 0=False): "))
restecg = int(input("Resting ECG (0-2): "))
thalach = float(input("Maximum Heart Rate Achieved: "))
exang = int(input("Exercise Induced Angina (1=Yes, 0=No): "))
oldpeak = float(input("ST Depression (oldpeak): "))
slope = int(input("Slope (0-2): "))
ca = int(input("Number of Major Vessels (0-3): "))
thal = int(input("Thal (1=Normal, 2=Fixed Defect, 3=Reversible Defect): "))


# Create DataFrame with SAME column names
new_patient = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

# Apply imputer
new_patient = pd.DataFrame(
    imputer.transform(new_patient),
    columns=X.columns
)

# Predict
prediction = rf.predict(new_patient)

print("\n===== Prediction Result =====")

if prediction[0] == 1:
    print("⚠️ High Risk of Hypertension")
else:
    print("✅ Low Risk of Hypertension")

print("\nProcess Finished Successfully ✅")