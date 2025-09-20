# imports
import pandas as pd

# file paths
file_paths = {
    "V Kohli": ["DATA/KOHLI_HOME.csv", "DATA/KOHLI_AWAY.csv"],
    "JE Root": ["DATA/ROOT_HOME.csv", "DATA/ROOT_AWAY.csv"],
    "SPD Smith": ["DATA/SMITH_HOME.csv", "DATA/SMITH_AWAY.csv"],
    "KS Williamson": ["DATA/KANE_HOME.csv", "DATA/KANE_AWAY.csv"]
}

# column conversion
int_columns = ["runs", "minutes", "balls", "fours", "sixes", "year", "inns"]
float_columns = ["strike_rate"]

# load clean merge function
def merge_home_away(player_name, file_list, home_nation):
    # Load home and away CSVs
    df_home = pd.read_csv(file_list[0])
    df_away = pd.read_csv(file_list[1])

    # drop unnamed column
    for df in [df_home, df_away]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    # standardize column name
    for df in [df_home, df_away]:
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # rename columns
    column_rename = {
        "player": "PlayerName",
        "runs": "Runs",
        "mins": "Minutes",
        "bf": "BallsFaced",
        "4s": "Fours",
        "6s": "Sixes",
        "sr": "StrikeRate",
        "inns": "MatchInning",
        "opposition": "Opposition",
        "ground": "Ground",
        "year": "Year",
        "venue": "Venue",
        "home_away": "Home/Away"
    }

    # apply renaming
    df_home.rename(columns=column_rename, inplace=True)
    df_away.rename(columns=column_rename, inplace=True)

    # add home and away as boolean
    df_home["Home/Away"] = 1
    df_away["Home/Away"] = 0

    # add nationality
    df_home["Nationality"] = home_nation
    df_away["Nationality"] = home_nation

    # merge home and away
    merged_df = pd.concat([df_home, df_away], ignore_index=True)

    # check data types
    for col in int_columns:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0).astype(int)

    for col in float_columns:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0)

    # verify opposition team names
    if "Opposition" in merged_df.columns:
        merged_df["Opposition"] = merged_df["Opposition"].str.lower().str.strip()

    # handle missing balls faced and minutes
    for col in ["Minutes", "BallsFaced"]:
        if col in merged_df.columns:
            mask = (merged_df[col] == 0) | (merged_df[col].isna())
            for idx in merged_df[mask].index:
                run_value = merged_df.at[idx, "Runs"]
                similar_rows = merged_df[merged_df["Runs"] == run_value][col]

                nearest_value = similar_rows[similar_rows > 0].median() # replace with non zero balls or minutes where similar runs

                if pd.isna(nearest_value):
                    nearest_value = merged_df[col][merged_df[col] > 0].median() # replace with closest value

                merged_df.at[idx, col] = nearest_value if not pd.isna(nearest_value) else 0

    merged_df.drop(columns=["Venue"], inplace=True)

    # sort data
    merged_df = merged_df.sort_values(
        by=["Year", "Opposition", "Home/Away", "Ground", "MatchInning"],
        ascending=[True, True, True, True, True]
    ).reset_index(drop=True)

    # save
    output_file = f"CLEANED_DATA/{player_name.replace(' ', '_').upper()}_MERGED.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned & sorted data for {player_name}")

# define each player
players_home_nation = {
    "V Kohli": "India",
    "JE Root": "England",
    "SPD Smith": "Australia",
    "KS Williamson": "New Zealand"
}

for player, files in file_paths.items():
    home_nation = players_home_nation[player]
    merge_home_away(player, files, home_nation)

print("\n✅ All 4 merged player files have been saved successfully!")