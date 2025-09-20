# imports
import joblib
import pandas as pd

# file paths
files = {
    "Joe Root": "CLEANED_DATA/JE_ROOT_MERGED.csv",
    "Kane Williamson": "CLEANED_DATA/KS_WILLIAMSON_MERGED.csv",
    "Steve Smith": "CLEANED_DATA/SPD_SMITH_MERGED.csv",
    "Virat Kohli": "CLEANED_DATA/V_KOHLI_MERGED.csv",
}

# calculate avg balls faced and strike rate
def get_player_averages(player_name, opposition):

    if player_name not in files:
        raise ValueError(f"Invalid player name: {player_name}")

    df = pd.read_csv(files[player_name])

    # filter for opposition
    opposition_data = df[df["Opposition"] == opposition]

    if not opposition_data.empty:
        avg_balls = opposition_data["BallsFaced"].mean()
        avg_sr = opposition_data["StrikeRate"].mean()
    else:
        avg_balls = df["BallsFaced"].mean()
        avg_sr = df["StrikeRate"].mean()

    return avg_balls, avg_sr


def load_model_and_predict(player_name, opposition, home_or_away, num_matches):

    model_path = f"MODELS/{player_name}_model.pkl"
    encoder_path = f"MODELS/{player_name}_label_encoder.pkl"

    # load model and encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    opposition_encoded = label_encoder.transform([opposition])[0]

    avg_balls, avg_sr = get_player_averages(player_name, opposition)

    # predict for future series
    total_predicted_runs = 0
    for match in range(num_matches):
        for innings in [1, 2]:
            new_data = pd.DataFrame({
                "Year": [2025],
                "Opposition": [opposition_encoded],
                "Home/Away": [home_or_away],
                "BallsFaced": [avg_balls],
                "StrikeRate": [avg_sr],
                "MatchInning": [innings]
            })
            predicted_runs = model.predict(new_data)[0]
            total_predicted_runs += predicted_runs
            total_predicted_runs = int(round(total_predicted_runs))
            opposition = opposition.upper()

    print(f"\n{player_name} - Predicted Runs vs {opposition} ({'Home' if home_or_away else 'Away'}) in {num_matches} matches: {total_predicted_runs}")
    return total_predicted_runs

# user input with validations
# list of players
available_players = {
    1: "Joe Root",
    2: "Kane Williamson",
    3: "Steve Smith",
    4: "Virat Kohli"
}
while True:
    print("Available players:", available_players)

    # get player input
    while True:
        try:
            choice = int(input("Enter the Number Corresponding to the Player: "))
            if choice in available_players:
                player_name = available_players[choice]
                break
            else:
                print("Please Enter a Number between 1 and 4.")
        except ValueError:
            print("Invalid Input. Please Enter a Valid Number.")

    # load data to get oppositions
    df = pd.read_csv(files[player_name])

    valid_oppositions = df['Opposition'].unique().tolist()

    # list oppositions
    print("Available Oppositions:")
    for team in sorted(valid_oppositions):
        print(f"- {team.title()}")

    # get opposition input
    while True:
        opposition = input("Enter Opposition Team: ").strip().lower()
        if opposition in valid_oppositions:
            break
        else:
            print("Invalid Opposition. Please Enter a Valid Team Name from the List Above.")

    # home or away
    while True:
        try:
            home_or_away = int(input("Enter 1 for Home, 0 for Away: "))
            if home_or_away in (0, 1):
                break
            else:
                print("Please Enter Only 1 for Home or 0 for Away.")
        except ValueError:
            print("Invalid Input. Please Enter a Number (0 or 1).")

    # number of matches in series
    while True:
        try:
            num_matches = int(input("Enter Number of Matches to Predict (1 to 5): "))
            if 1 <= num_matches <= 5:
                break
            else:
                print("Please Enter a Number Between 1 and 5.")
        except ValueError:
            print("Invalid Input. Please Enter a Valid Number.")

    load_model_and_predict(player_name, opposition, home_or_away, num_matches)

    while True:
        # ask if want to predict again
        again = input("\nWould you Like to Predict for Another Player? (yes/no): ").strip().lower()
        if again in ["yes", "no"]:
            break
        else:
            print("Please Enter 'yes' or 'no'.")
    if again == "no":
        print("Exiting FAB4 Predictor")
        break
