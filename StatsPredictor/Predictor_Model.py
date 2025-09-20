#all imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

#load datasets
d19 = pd.read_csv('CLEANED_DATA/D19_CLEAN.csv') #data until 2019
d24 = pd.read_csv('CLEANED_DATA/D24_CLEAN.csv') #data until 2024

#columns to scale
num_features = ['Matches', 'Innings', 'NotOut', 'HighestScore', 'Ducks',
            'Centuries', 'HalfCenturies', 'Average', 'FirstMatch',
            'LastMatch', 'CenturyConversion', 'FiftyPlusScorePercentage',
            'DuckPercentage', 'NotOutPercentage', 'CareerLength',
            'MatchesPerYear', 'CurrentPlayer']

target_column = ['Runs']

#one-hot encoding for countries
d19_encoded = pd.get_dummies(d19, columns=['Country'], drop_first=True)

#apply scaling
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()
d19_encoded[num_features] = scaler_features.fit_transform(d19_encoded[num_features])
d19_encoded[target_column] = scaler_target.fit_transform(d19_encoded[target_column])

#drop player name to train model on only numerical values
d19_encoded = d19_encoded.drop(columns=['PlayerName'])

#define features (X) and target (y)
X = d19_encoded.drop(columns=target_column)  #all features minus runs
y = d19_encoded[target_column]  #target = runs

#split data in train and test sets, 80 - 20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

#model evaluation
y_pred_scaled = model.predict(X_test)
y_test_actual = scaler_target.inverse_transform(y_test)
y_pred_actual = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1))

#evaluate model performance, displayed later
r2 = r2_score(y_test_actual, y_pred_actual)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = root_mean_squared_error(y_test_actual, y_pred_actual)

#output model metrics
print("\nGENERAL MODEL METRICS")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#cross-validation
cv_scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=10, scoring='r2')

#cross-validation results
print("\nCROSS-VALIDATION RESULTS")
print(f"Mean R² Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation of R² Scores: {cv_scores.std():.4f}")

#extract feature names
feature_names = X_train.columns
while True:
    #player name input, lower() is used standardize player name from input and data
    while True:
        player_name = input("Enter the Player's Name: ").strip().lower()
        #check in d19
        player_row = d19[d19['PlayerName'].str.lower() == player_name.lower()]
        #check in d24
        player_row_updated = d24[d24['PlayerName'].str.lower() == player_name.lower()]

        if not player_row.empty or not player_row_updated.empty:
            break
        else:
            print("Player Not Found in Dataset. Please Enter a Valid Player Name.")

    #player found in both d19 and d24
    if not player_row.empty and not player_row_updated.empty:
        #found in both
        print("Player Found in Both d19 and d24.")

        #extract player stats from d19
        player_stats = player_row.iloc[0].to_dict()

        #extract player stats from d24
        player_stats_updated = player_row_updated.iloc[0].to_dict()

        current_innings = player_stats['Innings']

        #expected innings input
        while True:
            try:
                expected_innings = int(input("Enter Total Number of Expected Innings: "))
                if expected_innings <= current_innings:
                    print("Expected Innings should be Greater than Current Innings.")
                    continue
                elif expected_innings > 500:
                    print("Enter a Realistic Value for Expected Innings")
                    continue
                break
            except ValueError:
                print("Invalid Input. Please Enter a Valid Number.")

        #d24 stats for display
        latest_name = player_stats_updated['PlayerName']
        latest_innings = player_stats_updated['Innings']
        latest_runs = player_stats_updated['Runs']
        latest_avg = player_stats_updated['Average']

        #predict for latest innings (d24 innings)
        latest_player_data = {
            "Matches": [player_stats_updated['Matches']],
            "Innings": [latest_innings],
            "NotOut": [player_stats_updated['NotOut']],
            "HighestScore": [player_stats_updated['HighestScore']],
            "Average": [latest_avg],
            "Centuries": [player_stats_updated['Centuries']],
            "HalfCenturies": [player_stats_updated['HalfCenturies']],
            "Ducks": [player_stats_updated['Ducks']],
            "FirstMatch": [player_stats_updated['FirstMatch']]
        }

        #convert to a dataframe
        latest_player_df = pd.DataFrame(latest_player_data)
        #re-order and match columns
        latest_player_df = latest_player_df.reindex(columns=feature_names, fill_value=0)
        #normalize the num features using the same scaler
        latest_player_df[num_features] = scaler_features.transform(latest_player_df[num_features])
        #predict using model, scaled output
        predicted_latest_scaled = model.predict(latest_player_df)[0]
        #convert scaled to actual (reverse scaling)
        predicted_latest_actual = scaler_target.inverse_transform([[predicted_latest_scaled]])[0][0]
        #round prediction
        predicted_latest_actual = int(round(predicted_latest_actual))

        #calculate expected innings and matches
        additional_innings = expected_innings - current_innings
        additional_matches = additional_innings // 2  #1 match = 2 innings

        #predict for expected innings
        future_player_data = {
            "Matches": [player_stats['Matches'] + additional_matches],
            "Innings": [expected_innings],
            "NotOut": [player_stats['NotOut'] + (additional_innings // 10)], #10% of new innings are not out
            "HighestScore": [player_stats['HighestScore']],
            "Average": [player_stats['Average']],
            "Centuries": [player_stats['Centuries'] + (additional_innings // 7)], #1 century in every 7 new innings
            "HalfCenturies": [player_stats['HalfCenturies'] + (additional_innings // 3)], #1 half century in every 3 new innings
            "Ducks": [player_stats['Ducks'] + (additional_innings // 20)], #1 duck in every 20 new innings
            "FirstMatch": [player_stats['FirstMatch']]
        }

        #convert to a dataframe
        future_player_df = pd.DataFrame(future_player_data)
        #re-order and match columns
        future_player_df = future_player_df.reindex(columns=feature_names, fill_value=0)
        #normalize the num features using the same scaler
        future_player_df[num_features] = scaler_features.transform(future_player_df[num_features])
        #predict using model, scaled output
        predicted_future_scaled = model.predict(future_player_df)[0]
        #convert scaled to actual (reverse scaling)
        predicted_future_actual = scaler_target.inverse_transform([[predicted_future_scaled]])[0][0]
        #round prediction
        predicted_future_actual = int(round(predicted_future_actual))

        #display player stats and predictions
        print("\n----- PLAYER STATS (Latest from d24) -----")
        print(f"PLAYER NAME: {latest_name.upper()}")
        print(f"CURRENT INNINGS: {latest_innings}")
        print(f"CURRENT RUNS: {latest_runs}")
        print(f"CURRENT AVERAGE: {latest_avg:.2f}")
        print(f"\nPredictions:")
        print(f"PREDICTED RUNS AFTER {latest_innings} INNINGS: {predicted_latest_actual}")
        print(f"PREDICTED RUNS AFTER {expected_innings} INNINGS: {predicted_future_actual}")

    #player found in only d19 and not in d24
    elif not player_row.empty and player_row_updated.empty:
        #found in d19 only
        print("Player Found in Only d19.")
        
        #extract player stats from d19
        player_stats = player_row.iloc[0].to_dict()

        current_innings = player_stats['Innings']

        #expected innings input
        while True:
            try:
                expected_innings = int(input("Enter Total Number of Expected Innings: "))
                if expected_innings <= current_innings:
                    print("Expected Innings should be Greater than Current Innings.")
                    continue
                elif expected_innings > 500:
                    print("Enter a Realistic Value for Expected Innings")
                    continue
                break
            except ValueError:
                print("Invalid Input. Please Enter a Valid Number.")

        #d19 stats for display
        latest_name = player_stats['PlayerName']
        latest_innings = player_stats['Innings']
        latest_runs = player_stats['Runs']
        latest_avg = player_stats['Average']        

        #predict for latest innings (d19 innings)
        latest_player_data = {
            "Matches": [player_stats['Matches']],
            "Innings": [latest_innings],
            "NotOut": [player_stats['NotOut']],
            "HighestScore": [player_stats['HighestScore']],
            "Average": [latest_avg],
            "Centuries": [player_stats['Centuries']],
            "HalfCenturies": [player_stats['HalfCenturies']],
            "Ducks": [player_stats['Ducks']],
            "FirstMatch": [player_stats['FirstMatch']]
        }

        #convert to a dataframe
        latest_player_df = pd.DataFrame(latest_player_data)
        #re-order and match columns
        latest_player_df = latest_player_df.reindex(columns=feature_names, fill_value=0)
        #normalize the num features using the same scaler
        latest_player_df[num_features] = scaler_features.transform(latest_player_df[num_features])
        #predict using model, scaled output
        predicted_latest_scaled = model.predict(latest_player_df)[0]
        #convert scaled to actual (reverse scaling)
        predicted_latest_actual = scaler_target.inverse_transform([[predicted_latest_scaled]])[0][0]
        #round prediction
        predicted_latest_actual = int(round(predicted_latest_actual))

        #calculate expected innings and matches
        additional_innings = expected_innings - current_innings
        additional_matches = additional_innings // 2  #1 match = 2 innings

        #predict for expected innings
        future_player_data = {
            "Matches": [player_stats['Matches'] + additional_matches],
            "Innings": [expected_innings],
            "NotOut": [player_stats['NotOut'] + (additional_innings // 10)], #10% of new innings are not out
            "HighestScore": [player_stats['HighestScore']],
            "Average": [player_stats['Average']],
            "Centuries": [player_stats['Centuries'] + (additional_innings // 7)], #1 century in every 7 new innings
            "HalfCenturies": [player_stats['HalfCenturies'] + (additional_innings // 3)], #1 half century in every 3 new innings
            "Ducks": [player_stats['Ducks'] + (additional_innings // 20)], #1 duck in every 20 new innings
            "FirstMatch": [player_stats['FirstMatch']]
        }

        #convert to a dataframe
        future_player_df = pd.DataFrame(future_player_data)
        #re-order and match columns
        future_player_df = future_player_df.reindex(columns=feature_names, fill_value=0)
        #normalize the num features using the same scaler
        future_player_df[num_features] = scaler_features.transform(future_player_df[num_features])
        #predict using model, scaled output
        predicted_future_scaled = model.predict(future_player_df)[0]
        #convert scaled to actual (reverse scaling)
        predicted_future_actual = scaler_target.inverse_transform([[predicted_future_scaled]])[0][0]
        #round prediction
        predicted_future_actual = int(round(predicted_future_actual))

        #display player stats and predictions
        print("\n----- PLAYER STATS (from d19) -----")
        print(f"PLAYER NAME: {latest_name.upper()}")
        print(f"CURRENT INNINGS: {latest_innings}")
        print(f"CURRENT RUNS: {latest_runs}")
        print(f"CURRENT AVERAGE: {latest_avg:.2f}")
        print(f"\nPredictions:")
        print(f"PREDICTED RUNS AFTER {latest_innings} INNINGS: {predicted_latest_actual}")
        print(f"PREDICTED RUNS AFTER {expected_innings} INNINGS: {predicted_future_actual}")

    #player found in only d24 and not in d19
    elif not player_row_updated.empty and player_row.empty:
        #found in d24 only
        print("Player Found in Only d24.")
        
        #extract player stats from d24
        player_stats_updated = player_row_updated.iloc[0].to_dict()

        current_innings = player_stats_updated['Innings']

        #expected innings input
        while True:
            try:
                expected_innings = int(input("Enter Total Number of Expected Innings: "))
                if expected_innings <= current_innings:
                    print("Expected Innings should be Greater than Current Innings.")
                    continue
                elif expected_innings > 500:
                    print("Enter a Realistic Value for Expected Innings")
                    continue
                break
            except ValueError:
                print("Invalid Input. Please Enter a Valid Number.")

        #d24 stats for display
        latest_name = player_stats_updated['PlayerName']
        latest_innings = player_stats_updated['Innings']
        latest_runs = player_stats_updated['Runs']
        latest_avg = player_stats_updated['Average']

        #predict for latest innings (d24 innings)
        latest_player_data = {
            "Matches": [player_stats_updated['Matches']],
            "Innings": [latest_innings],
            "NotOut": [player_stats_updated['NotOut']],
            "HighestScore": [player_stats_updated['HighestScore']],
            "Average": [latest_avg],
            "Centuries": [player_stats_updated['Centuries']],
            "HalfCenturies": [player_stats_updated['HalfCenturies']],
            "Ducks": [player_stats_updated['Ducks']],
            "FirstMatch": [player_stats_updated['FirstMatch']]
        }

        #convert to a dataframe
        latest_player_df = pd.DataFrame(latest_player_data)
        #re-order and match columns
        latest_player_df = latest_player_df.reindex(columns=feature_names, fill_value=0)
        #normalize the num features using the same scaler
        latest_player_df[num_features] = scaler_features.transform(latest_player_df[num_features])
        #predict using model, scaled output
        predicted_latest_scaled = model.predict(latest_player_df)[0]
        #convert scaled to actual (reverse scaling)
        predicted_latest_actual = scaler_target.inverse_transform([[predicted_latest_scaled]])[0][0]
        #round prediction
        predicted_latest_actual = int(round(predicted_latest_actual))

        #calculate expected innings and matches
        additional_innings = expected_innings - current_innings
        additional_matches = additional_innings // 2  #1 match = 2 innings

        #predict for expected innings
        future_player_data = {
            "Matches": [player_stats_updated['Matches'] + additional_matches],
            "Innings": [expected_innings],
            "NotOut": [player_stats_updated['NotOut'] + (additional_innings // 10)], #10% of new innings are not out
            "HighestScore": [player_stats_updated['HighestScore']],
            "Average": [player_stats_updated['Average']],
            "Centuries": [player_stats_updated['Centuries'] + (additional_innings // 7)], #1 century in every 7 new innings
            "HalfCenturies": [player_stats_updated['HalfCenturies'] + (additional_innings // 3)], #1 half century in every 3 new innings
            "Ducks": [player_stats_updated['Ducks'] + (additional_innings // 20)], #1 duck in every 20 new innings
            "FirstMatch": [player_stats_updated['FirstMatch']]
        }

        #convert to a dataframe
        future_player_df = pd.DataFrame(future_player_data)
        #re-order and match columns
        future_player_df = future_player_df.reindex(columns=feature_names, fill_value=0)
        #normalize the num features using the same scaler
        future_player_df[num_features] = scaler_features.transform(future_player_df[num_features])
        #predict using model, scaled output
        predicted_future_scaled = model.predict(future_player_df)[0]
        #convert scaled to actual (reverse scaling)
        predicted_future_actual = scaler_target.inverse_transform([[predicted_future_scaled]])[0][0]
        #round prediction
        predicted_future_actual = int(round(predicted_future_actual))

        #display player stats and predictions
        print("\n----- PLAYER STATS (from d24) -----")
        print(f"PLAYER NAME: {latest_name.upper()}")
        print(f"CURRENT INNINGS: {latest_innings}")
        print(f"CURRENT RUNS: {latest_runs}")
        print(f"CURRENT AVERAGE: {latest_avg:.2f}")
        print(f"\nPredictions:")
        print(f"PREDICTED RUNS AFTER {latest_innings} INNINGS: {predicted_latest_actual}")
        print(f"PREDICTED RUNS AFTER {expected_innings} INNINGS: {predicted_future_actual}")

    #another prediction
    while True:
        another = input("Do You Want to Predict Stats for Another Player? (yes/no): ").strip().lower()
        if another in ['yes', 'no']:
            break
        else:
            print("Please Enter 'yes' or 'no'.")
    if another == 'no':
        print("Exiting Predictor Model")
        break