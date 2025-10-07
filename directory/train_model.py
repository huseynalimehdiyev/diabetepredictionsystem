import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Dataset
df = pd.read_csv('diabetes.csv')

#Medianla əvəzləmə
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zeros:
    df[col] = df[col].replace(0, df[col].median())


X = df.drop("Outcome", axis=1)

y = df["Outcome"]

# Train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelin qurulmasi ve öyredilmsi
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Modelin fayla yazilması
joblib.dump(model, "diabetes_model.pkl")
print("✅ Model uğurla 'diabetes_model.pkl' olaraq saxlanıldı.")
