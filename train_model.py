import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Dummy dataset (replace this with your real dataset)
# Columns: N, P, K, Temp, Humidity, Fertility
data = {
    "N": np.random.randint(0, 300, 100),
    "P": np.random.randint(0, 300, 100),
    "K": np.random.randint(0, 300, 100),
    "Temperature": np.random.randint(10, 50, 100),
    "Humidity": np.random.randint(0, 100, 100),
    "Fertility": np.random.choice(["Low", "Medium", "High"], 100)
}

df = pd.DataFrame(data) 

# Features and labels
X = df[["N", "P", "K", "Temperature", "Humidity"]]
y = df["Fertility"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy to file
with open("model_accuracy.txt", "w") as f:
    f.write(str(accuracy))

# Save feature importances
if hasattr(model, "feature_importances_"):
    feature_names = ['N', 'P', 'K', 'Temp', 'Humidity']

    importance_df = pd.DataFrame({
        "Feature": feature_names,  # e.g. ['N', 'P', 'K', 'Temp', 'Humidity']
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    importance_df.to_csv("feature_importance.csv", index=False)

# Save predictions to CSV for Streamlit dashboard
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
results_df.to_csv("ml_results.csv", index=False)

# Save model
with open("small_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'small_model.pkl'")
print("📄 Predictions saved to 'ml_results.csv'")