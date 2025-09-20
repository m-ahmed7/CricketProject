#imports
import pandas as pd
import re

#load data
d19 = pd.read_csv('DATA/STATS_2019.csv', encoding='ISO-8859-1')
d24 = pd.read_csv('DATA/STATS_2024.csv', encoding='ISO-8859-1')

#rename columns
d19 = d19.rename(columns={"Player": "PlayerName", "Span": "CareerSpan", "Mat": "Matches", "Inn": "Innings",
                "NO": "NotOut", "HS": "HighestScore", "Avg": "Average", "100": "Centuries", 
                "50": "HalfCenturies", "0": "Ducks"})

d24 = d24.rename(columns={"name": "PlayerName", "span": "CareerSpan", "matches": "Matches", "innings": "Innings",
                "not_out": "NotOut", "runs": "Runs", "highest_score": "HighestScore", "average": "Average", 
                "century": "Centuries", "half_century": "HalfCenturies", "ducks": "Ducks", "country": "Country"})

#drop players who have not batted yet
d19.drop(d19[d19['Innings'] == '-'].index, inplace=True)
d19.drop(d19[d19['Runs'].isin(['-', '0'])].index, inplace=True)

d24.drop(d24[d24['Innings'] == '-'].index, inplace=True)
d24.drop(d24[d24['Runs'].isin(['-', '0'])].index, inplace=True)

#where batter is not out, they dont have average so runs = average
d19.loc[d19["Average"] == "-", "Average"] = d19["Runs"]

d24.loc[d24["Average"] == "-", "Average"] = d24["Runs"]

#remove * from high score
#indicates not out in highest score innings but not needed for analysis
d19['HighestScore'] = d19['HighestScore'].str.replace('*', '', regex=False)

d24['HighestScore'] = d24['HighestScore'].astype(str).str.replace('*', '', regex=False)

#create country as separate column, extract it from player name
d19["Country"] = d19["PlayerName"].apply(lambda x: re.search(r"\((.*?)\)", x).group(1) if "(" in x else None)
d19["PlayerName"] = d19["PlayerName"].str.replace(r"\(.*?\)", "", regex=True).str.strip()
#remove ICC and restrict length of country to 3 characters
d19["Country"] = d19["Country"].str.replace(r"(^ICC/|/ICC$)", "", regex=True).str.strip()
#drop dual national players and a player with nationality 3
d19.drop(d19[d19["Country"].str.contains("/", regex=True, na=False)].index, inplace=True)
d19.drop(d19[d19["Country"] == "3"].index, inplace=True)
#fix India and Bangladesh 
d19["Country"] = d19["Country"].replace("INDIA", "IND")
d19["Country"] = d19["Country"].replace("BDESH", "BAN")

#extract first and last match year
d19[['FirstMatch', 'LastMatch']] = d19['CareerSpan'].str.split('-', expand=True)
d19.drop(columns=['CareerSpan'], inplace=True)

d24[['FirstMatch', 'LastMatch']] = d24['CareerSpan'].str.split('-', expand=True)
d24.drop(columns=['CareerSpan'], inplace=True)

#convert columns to int and avg to float
cols_to_int = ["Matches", "Innings", "NotOut", "Runs", "HighestScore", "Centuries",
                "HalfCenturies", "Ducks", "FirstMatch", "LastMatch"]
d19[cols_to_int] = d19[cols_to_int].apply(pd.to_numeric, errors="coerce").astype("Int64")
d19["Average"] = pd.to_numeric(d19["Average"], errors="coerce").astype(float)

d24[cols_to_int] = d24[cols_to_int].apply(pd.to_numeric, errors="coerce").astype("Int64")
d24["Average"] = pd.to_numeric(d24["Average"], errors="coerce").astype(float)

#adding new calculated columns
#rate of converting 50s into 100s
#calculates % of ducks, not outs, and 50+ scores
d19["CenturyConversion"] = d19.apply(
    lambda row: (row["Centuries"] / (row["Centuries"] + row["HalfCenturies"])) if row["Centuries"] > 0 else 0, axis=1
).round(4)
d19["DuckPercentage"] = (d19["Ducks"] / d19["Innings"]).round(4)
d19["NotOutPercentage"] = (d19["NotOut"] / d19["Innings"]).round(4)
d19["FiftyPlusScorePercentage"] = ((d19["HalfCenturies"] + d19["Centuries"]) / d19["Innings"]).round(4)
d19["FiftyPlusScorePercentage"] = d19.apply(
    lambda row: 0 if row["Innings"] == 0 else row["FiftyPlusScorePercentage"], axis=1
)

d24["CenturyConversion"] = d24.apply(
    lambda row: (row["Centuries"] / (row["Centuries"] + row["HalfCenturies"])) if row["Centuries"] > 0 else 0, axis=1
).round(4)
d24["DuckPercentage"] = (d24["Ducks"] / d24["Innings"]).round(4)
d24["NotOutPercentage"] = (d24["NotOut"] / d24["Innings"]).round(4)
d24["FiftyPlusScorePercentage"] = ((d24["HalfCenturies"] + d24["Centuries"]) / d24["Innings"]).round(4)
d24["FiftyPlusScorePercentage"] = d24.apply(
    lambda row: 0 if row["Innings"] == 0 else row["FiftyPlusScorePercentage"], axis=1
)

#career length, if first and last match same year then 1
d19["CareerLength"] = d19["LastMatch"] - d19["FirstMatch"]
d19["CareerLength"] = d19["CareerLength"].apply(lambda x: 1 if x == 0 else x)

d24["CareerLength"] = d24["LastMatch"] - d24["FirstMatch"]
d24["CareerLength"] = d24["CareerLength"].apply(lambda x: 1 if x == 0 else x)

#matches per year
d19["MatchesPerYear"] = (d19["Matches"] / d19["CareerLength"]).round(2)

d24["MatchesPerYear"] = (d24["Matches"] / d24["CareerLength"]).round(2)

#current and retired players,  as per data sets
d19["CurrentPlayer"] = (d19["LastMatch"] == 2019).astype(int)

d24["CurrentPlayer"] = (d24["LastMatch"] == 2024).astype(int)

#final check for null values
print(d19.isnull().sum().sum())
print(d24.isnull().sum().sum())

#save cleaned data to csv files
output = "CLEANED_DATA/D19_CLEAN.csv"
d19.to_csv(output, index=False)
output = f"CLEANED_DATA/D24_CLEAN.csv"
d24.to_csv(output, index=False)