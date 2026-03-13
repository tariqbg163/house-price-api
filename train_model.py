import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Create dataset
data = {
    "size":[500,700,900,1100,1300,1500],
    "price":[100000,120000,140000,160000,180000,200000]
}

df = pd.DataFrame(data)

# Feature
X = df[["size"]]

# Target
y = df["price"]

# Create model
model = LinearRegression()

# Train model
model.fit(X,y)

# Save model
joblib.dump(model,"model.pkl")

print("Model saved")