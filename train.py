import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# =========================================================
# 1. LOAD DATA
# =========================================================
FILE_PATH = "used_cars.csv"   # <- replace with your CSV file name if different

df = pd.read_csv(FILE_PATH)

# print("Shape before cleaning:", df.shape)
# print("\nColumns:")
# print(df.columns.tolist())
# print("\nMissing values:")
# print(df.isnull().sum())


# =========================================================
# 2. CLEAN PRICE COLUMN
# Example: "$10,300" -> 10300
# =========================================================
df["price"] = (
    df["price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["price"] = pd.to_numeric(df["price"], errors="coerce")


# =========================================================
# 3. CLEAN MILAGE COLUMN
# Example: "51,000 mi." -> 51000
# =========================================================
df["milage"] = (
    df["milage"]
    .astype(str)
    .str.replace("mi.", "", regex=False)
    .str.replace("mi", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["milage"] = pd.to_numeric(df["milage"], errors="coerce")


# =========================================================
# 4. DROP DUPLICATES
# =========================================================
df = df.drop_duplicates()


# =========================================================
# 5. DROP ROWS WHERE TARGET IS MISSING
# =========================================================
df = df.dropna(subset=["price"])


# =========================================================
# 6. FEATURE ENGINEERING
# Create car_age from model_year
# =========================================================
current_year = 2026
df["car_age"] = current_year - df["model_year"]


# =========================================================
# 7. GROUP RARE BRANDS INTO "Other"
# Keep brands that appear at least 5 times
# =========================================================
brand_counts = df["brand"].value_counts()
rare_brands = brand_counts[brand_counts < 5].index
df["brand"] = df["brand"].replace(rare_brands, "Other")


# =========================================================
# 8. SIMPLIFY ACCIDENT COLUMN
# Convert messy accident text into fewer categories
# =========================================================
def simplify_accident(x):
    x = str(x).strip().lower()

    if x in ["nan", "none", "no accidents reported", "no accident", "none reported"]:
        return "No Accident"

    if "none reported" in x or "no accident" in x:
        return "No Accident"

    if "accident" in x or "damage" in x:
        return "Accident Reported"

    return "Unknown"

df["accident"] = df["accident"].apply(simplify_accident)


# =========================================================
# 9. CLEAN clean_title COLUMN
# =========================================================
def simplify_clean_title(x):
    x = str(x).strip().lower()

    if x in ["yes", "true", "1"]:
        return "Yes"
    elif x in ["no", "false", "0"]:
        return "No"
    else:
        return "Unknown"

df["clean_title"] = df["clean_title"].apply(simplify_clean_title)


# =========================================================
# 10. REMOVE EXTREME PRICE OUTLIERS
# Remove bottom 1% and top 1%
# =========================================================
q1 = df["price"].quantile(0.01)
q99 = df["price"].quantile(0.99)

df = df[(df["price"] >= q1) & (df["price"] <= q99)]

# print("\nShape after cleaning + outlier removal:", df.shape)


# =========================================================
# 11. CHOOSE FEATURES
# IMPORTANT:
# We are dropping high-cardinality / noisy columns for now:
# - model
# - engine
# - ext_col
# - int_col
#
# This usually gives a cleaner first model.
# =========================================================
selected_cols = [
    "brand",
    "fuel_type",
    "transmission",
    "accident",
    "clean_title",
    "milage",
    "car_age",
    "price"
]

df = df[selected_cols].copy()


# =========================================================
# 12. SPLIT FEATURES / TARGET
# =========================================================
X = df.drop("price", axis=1)
y = df["price"]


# =========================================================
# 13. NUMERIC AND CATEGORICAL COLUMNS
# =========================================================
num_cols = ["milage", "car_age"]

cat_cols = [
    "brand",
    "fuel_type",
    "transmission",
    "accident",
    "clean_title"
]


# =========================================================
# 14. TRAIN-TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================================
# 15. PREPROCESSING PIPELINES
# =========================================================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


# =========================================================
# 16. BUILD MODELS
# =========================================================
lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

gb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ))
])


# =========================================================
# 17. EVALUATION FUNCTION
# =========================================================
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n===== {model_name} =====")
    print("MAE : ", round(mae, 2))
    print("RMSE: ", round(rmse, 2))
    print("R2  : ", round(r2, 4))

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# =========================================================
# 18. TRAIN MODELS
# =========================================================
print("\nTraining Linear Regression...")
lr_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

print("Training Gradient Boosting...")
gb_model.fit(X_train, y_train)


# =========================================================
# 19. PREDICT
# =========================================================
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)


# =========================================================
# 20. EVALUATE
# =========================================================
results = []

results.append(evaluate_model(y_test, lr_preds, "Linear Regression"))
results.append(evaluate_model(y_test, rf_preds, "Random Forest"))
results.append(evaluate_model(y_test, gb_preds, "Gradient Boosting"))

print("\nAll results:")
for result in results:
    print(result['model'], "-> Accuracy:", round(result['r2'] * 100, 2),"%")


# =========================================================
# 21. FIND BEST MODEL (based on highest R2)
# =========================================================
best_result = max(results, key=lambda x: x["r2"])
best_model_name = best_result["model"]

if best_model_name == "Linear Regression":
    best_model = lr_model
elif best_model_name == "Random Forest":
    best_model = rf_model
else:
    best_model = gb_model

print("\nBest model:", best_model_name)
print("Best R2:", round(best_result["r2"], 4))


# =========================================================
# 22. SAVE BEST MODEL
# =========================================================
joblib.dump(best_model, "model.pkl")
print("\nBest model saved as: model.pkl")


# =========================================================
# 23. OPTIONAL: TEST ON A SAMPLE INPUT
# =========================================================
sample_car = pd.DataFrame({
    "brand": ["Ford"],
    "fuel_type": ["Gasoline"],
    "transmission": ["Automatic"],
    "accident": ["No Accident"],
    "clean_title": ["Yes"],
    "milage": [45000],
    "car_age": [4]
})

sample_pred = best_model.predict(sample_car)
print("\nSample prediction:", round(sample_pred[0], 2))

