"""
House Price Predictor — Ridge/RandomForest/GradientBoosting ensemble
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib, warnings
warnings.filterwarnings("ignore")

def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    print(f"Dataset: {df.shape[0]} houses, {df.shape[1]-1} features")
    return df

def train(df):
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
    }

    best_model, best_r2 = None, -999
    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        print(f"  {name}: R2={cv.mean():.4f} (+/- {cv.std():.4f})")
        if cv.mean() > best_r2:
            best_r2 = cv.mean()
            best_model = pipe

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"\nTest R2:   {r2_score(y_test, y_pred):.4f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
    print(f"Test MAE:  {mean_absolute_error(y_test, y_pred):.4f}")

    # Feature importance (Random Forest step)
    rf = best_model.named_steps["model"]
    if hasattr(rf, "feature_importances_"):
        fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop Features:"); print(fi.head(5))

    joblib.dump(best_model, "house_price_model.pkl")
    print("\nModel saved: house_price_model.pkl")
    return best_model

def predict(model, features):
    """Predict price for a single house."""
    arr = np.array(features).reshape(1, -1)
    price = model.predict(arr)[0]
    print(f"Predicted price: ${price * 100000:.0f}")
    return price

def main():
    print("=" * 60)
    print("  House Price Predictor")
    print("=" * 60)
    df = load_data()
    model = train(df)
    # Example prediction (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Lon)
    print("\nExample prediction:")
    predict(model, [8.0, 20.0, 6.0, 1.0, 500.0, 2.5, 37.5, -122.0])

if __name__ == "__main__":
    main()
