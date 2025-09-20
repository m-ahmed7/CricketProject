# imports
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# file paths
files = {
    "Joe Root": "CLEANED_DATA/JE_ROOT_MERGED.csv",
    "Kane Williamson": "CLEANED_DATA/KS_WILLIAMSON_MERGED.csv",
    "Steve Smith": "CLEANED_DATA/SPD_SMITH_MERGED.csv",
    "Virat Kohli": "CLEANED_DATA/V_KOHLI_MERGED.csv",
}

# store models
model_dir = "MODELS/"
os.makedirs(model_dir, exist_ok=True)

def train_and_save_models():
    "Trains RandomForest models for each player and saves them."
    for player_name, file_path in files.items():

        df = pd.read_csv(file_path)

        # encode oppositions
        label_encoder = LabelEncoder()
        df["Opposition"] = label_encoder.fit_transform(df["Opposition"])

        # save the encoder to decode later
        joblib.dump(label_encoder, f"{model_dir}{player_name}_label_encoder.pkl")

        # define features and target
        X = df[["Year", "Opposition", "Home/Away", "BallsFaced", "StrikeRate", "MatchInning"]]
        y = df["Runs"]

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # save the model
        joblib.dump(model, f"{model_dir}{player_name}_model.pkl")

        # model evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model trained for {player_name} - MAE: {mae:.2f}, R²: {r2:.2f}")

# run to train and save models
train_and_save_models()