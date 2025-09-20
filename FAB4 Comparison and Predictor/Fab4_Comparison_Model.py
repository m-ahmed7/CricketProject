# imports
import pandas as pd
import matplotlib.pyplot as plt

# file paths
files = {
    "Joe Root": "CLEANED_DATA/JE_ROOT_MERGED.csv",
    "Kane Williamson": "CLEANED_DATA/KS_WILLIAMSON_MERGED.csv",
    "Steve Smith": "CLEANED_DATA/SPD_SMITH_MERGED.csv",
    "Virat Kohli": "CLEANED_DATA/V_KOHLI_MERGED.csv",
}

# metrics list
metrics_list = [
    "Total Runs", "Batting Average", "Strike Rate", "First Innings Average",
    "Second Innings Average", "Third Innings Average", "Fourth Innings Average",
    "Fifty Plus Scores", "Half Centuries", "Centuries", "Double Centuries",
    "Boundaries (4s + 6s)", "Peak Year"
]

print("Choose the Metrics you want to Compare (comma-separated):")
for i, metric in enumerate(metrics_list, 1):
    print(f"{i}. {metric}")

# check valid input
while True:
    selected_input = input("Enter Metric Numbers: ").split(',')
    try:
        selected_metrics = []
        for choice in selected_input:
            choice = choice.strip()
            if not choice.isdigit():
                raise ValueError("Non-Numeric Input Detected.")
            idx = int(choice)
            if idx < 1 or idx > len(metrics_list):
                raise ValueError(f"Choice {idx} is Out of Valid Range.")
            selected_metrics.append(metrics_list[idx - 1])
        break
    except ValueError as ve:
        print(f"Invalid Input: {ve}. Please Enter Valid Numbers from the List.")

# select filters
filters_list = ["Filter by Home/Away", "Filter by Year Range", "Filter by Opposition"]
print("Choose Filters to Apply (comma-separated, enter 0 for none):")
for i, fltr in enumerate(filters_list, 1):
    print(f"{i}. {fltr}")

# check valid input
while True:
    selected_filters = input("Enter Filter Numbers: ").split(',')
    selected_filters = [choice.strip() for choice in selected_filters if choice.strip().isdigit()]

    if all(choice in ('0', '1', '2', '3') for choice in selected_filters):
        if '0' in selected_filters and len(selected_filters) == 1:
            selected_filters = []
        break
    else:
        print("Invalid Input. Please enter Valid Filter Numbers from the List (comma-separated).")

home_away, start_year, end_year, opposition = None, None, None, None
if '1' in selected_filters:
    while True:
        try:
            home_away = int(input("Choose Home/Away (1 for Home, 0 for Away): "))
            if home_away in (0, 1):
                break
            else:
                print("Please Enter Only 1 for Home or 0 for Away.")
        except ValueError:
            print("Invalid Input. Please Enter a Number (0 or 1).")
if '2' in selected_filters:
    while True:
        try:
            start_year = int(input("Enter Start Year: "))
            end_year = int(input("Enter End Year: "))
            if start_year <= end_year:
                break
            else:
                print("Start Year Cannot beGreater than End Year. Please Try Again.")
        except ValueError:
            print("Invalid Input. Please Enter Valid Years (e.g., 2015).")
if '3' in selected_filters:
    opposition = input("Enter Opposition Team: ").strip().lower()

# fixed max values assigned based on available data to assign scores
MAX_VALUES = {
    "Total Runs": 10000,
    "Batting Average": 100,
    "Strike Rate": 70,
    "Fifty Plus Scores": 100,
    "Half Centuries": 75,
    "Centuries": 50,
    "Double Centuries": 10,
    "Boundaries (4s + 6s)": 1000,
    "Peak Year": 2000
}

# normalize metrics
def normalize(value, max_val):
    return round(10 * (value / max_val), 2) if max_val else 0

# normalize batting average
def normalize_batting_avg(avg):
    return round((avg / 100) * 10, 2)

# assign filters and extract records as required
all_results = {}
for player, file in files.items():
    df = pd.read_csv(file)
    if '1' in selected_filters:
        df = df[df['Home/Away'] == home_away]
    if '2' in selected_filters:
        df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    if '3' in selected_filters:
        df = df[df['Opposition'].str.contains(opposition, case=False)]
    
    metrics_results = {}
    
    # calculate metrics
    if "Total Runs" in selected_metrics:
        metrics_results['Total Runs'] = int(df['Runs'].sum())
    if "Batting Average" in selected_metrics:
        avg = round(df['Runs'].sum() / df['MatchInning'].count(), 2) if df['MatchInning'].count() > 0 else 0
        metrics_results['Batting Average'] = avg
    if "Strike Rate" in selected_metrics:
        metrics_results['Strike Rate'] = round((df['Runs'].sum() / df['BallsFaced'].sum()) * 100, 2) if df['BallsFaced'].sum() > 0 else 0
    innings = ["First", "Second", "Third", "Fourth"]
    for i, inning in enumerate(innings, 1):
        if f"{inning} Innings Average" in selected_metrics:
            df_inning = df[df['MatchInning'] == i]
            metrics_results[f"{inning} Innings Average"] = round(df_inning['Runs'].mean(), 2) if not df_inning.empty else 0
    if "Fifty Plus Scores" in selected_metrics:
        metrics_results['Fifty Plus Scores'] = int(len(df[df['Runs'] >= 50]))
    if "Half Centuries" in selected_metrics:
        metrics_results['Half Centuries'] = int(len(df[(df['Runs'] >= 50) & (df['Runs'] < 100)]))
    if "Centuries" in selected_metrics:
        metrics_results['Centuries'] = int(len(df[(df['Runs'] >= 100) & (df['Runs'] < 200)]))
    if "Double Centuries" in selected_metrics:
        metrics_results['Double Centuries'] = int(len(df[df['Runs'] >= 200]))
    if "Boundaries (4s + 6s)" in selected_metrics:
        metrics_results['Boundaries (4s + 6s)'] = int(df['Fours'].sum() + df['Sixes'].sum())
    if "Peak Year" in selected_metrics:
        peak_df = df.groupby('Year')['Runs'].sum()
        if not peak_df.empty:
            peak_year = peak_df.idxmax()
            peak_runs = peak_df.max()
            peak_avg = round(df[df['Year'] == peak_year]['Runs'].mean(), 2)
            metrics_results['Peak Year'] = f"{peak_year} (Runs: {peak_runs}, Avg: {peak_avg})"

    all_results[player] = metrics_results

# assign scores and calculate rankings
player_scores = {}
for player, metrics in all_results.items():
    total_score = 0
    scores = {}

    scored_metrics = [m for m in selected_metrics if m != "Peak Year"]

    for metric_name, value in metrics.items():
        if metric_name == "Peak Year":
            continue  # deal with peak year separately
        if metric_name in ["Batting Average", "First Innings Average", "Second Innings Average", "Third Innings Average", "Fourth Innings Average"]:
            score = normalize_batting_avg(value)
        else:
            # Filter only numeric values from the metrics
            numeric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
            max_val = MAX_VALUES.get(metric_name, max(numeric_values))  # Use only numeric values
            score = normalize(value, max_val)
        total_score += score
        scores[metric_name + " (Score)"] = score

    # Store the normalized scores
    all_results[player].update(scores)

    # Calculate the final score by averaging the normalized scores
    player_scores[player] = round(total_score / len(scored_metrics), 2)

# rank players
ranked_players = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)

# display rankings
print("\nRankings Based on Selected Metrics:")
for rank, (player, score) in enumerate(ranked_players, start=1):
    print(f"{rank}. {player} - Final Score: {score}/10")

# yes no prompt
def get_yes_no(prompt):
    while True:
        ans = input(prompt).strip().lower()
        if ans in ('yes', 'no'):
            return ans
        print("Please Enter 'yes' or 'no'.")

# ask if want to see detailed stats
show_details = get_yes_no("Do You Want to see the Detailed Stats? (yes/no): ")

# display detailed stats if needed
if show_details == 'yes':
    print("\nDetailed Player Metrics with Scores:")
    for player in ranked_players:
        name = player[0]
        print(f"\n{name}:")
        for metric, value in all_results[name].items():
            print(f"{metric}: {value}")
            
# color assignment
player_colors = {
    "Virat Kohli": 'blue',
    "Joe Root": 'red',
    "Steve Smith": 'yellow',
    "Kane Williamson": 'grey',
}

# plotting graphs
def plot_metrics():

    # graph total runs
    if "Total Runs" in selected_metrics:
        plt.figure(figsize=(10, 6))
        for player in all_results:
            total_runs = all_results[player].get("Total Runs", 0)
            plt.bar(player, total_runs, color=player_colors.get(player, 'gray'))
        plt.title("Total Runs")
        plt.xlabel("Players")
        plt.ylabel("Total Runs")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    innings_metrics = ["Batting Average", "First Innings Average", "Second Innings Average", "Third Innings Average", "Fourth Innings Average"]
    selected_innings_metrics = [metric for metric in innings_metrics if metric in selected_metrics]
    if selected_innings_metrics:
        plt.figure(figsize=(10, 6))
        for player in all_results:
            # Filter batting averages based on the selected metrics
            batting_averages = [all_results[player].get(metric, 0) for metric in selected_innings_metrics]
            plt.bar([f"{player} - {metric.replace(' Innings Average', '')}" for metric in selected_innings_metrics], 
                    batting_averages, color=player_colors.get(player, 'gray'))
        plt.title("Batting Averages")
        plt.xlabel("Player - Innings")
        plt.ylabel("Average")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    # graph strike rate
    if "Strike Rate" in selected_metrics:
        plt.figure(figsize=(10, 6))
        for player in all_results:
            strike_rate = all_results[player].get("Strike Rate", 0)
            plt.bar(player, strike_rate, color=player_colors.get(player, 'gray'))
        plt.title("Strike Rate")
        plt.xlabel("Players")
        plt.ylabel("Strike Rate")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # milestones graph
    if any(metric in selected_metrics for metric in ["Fifty Plus Scores", "Half Centuries", "Centuries", "Double Centuries"]):
        if "Fifty Plus Scores" in selected_metrics:
            plt.figure(figsize=(10, 6))
            for player in all_results:
                plt.bar(player, all_results[player].get("Fifty Plus Scores", 0), color=player_colors.get(player, 'gray'))
            plt.title("Fifty Plus Scores")
            plt.xlabel("Players")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        if "Half Centuries" in selected_metrics:
            plt.figure(figsize=(10, 6))
            for player in all_results:
                plt.bar(player, all_results[player].get("Half Centuries", 0), color=player_colors.get(player, 'gray'))
            plt.title("Half Centuries")
            plt.xlabel("Players")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        if "Centuries" in selected_metrics:
            plt.figure(figsize=(10, 6))
            for player in all_results:
                plt.bar(player, all_results[player].get("Centuries", 0), color=player_colors.get(player, 'gray'))
            plt.title("Centuries")
            plt.xlabel("Players")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        if "Double Centuries" in selected_metrics:
            plt.figure(figsize=(10, 6))
            for player in all_results:
                plt.bar(player, all_results[player].get("Double Centuries", 0), color=player_colors.get(player, 'gray'))
            plt.title("Double Centuries")
            plt.xlabel("Players")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # graph boundaries
    if "Boundaries (4s + 6s)" in selected_metrics:
        plt.figure(figsize=(10, 6))
        for player in all_results:
                boundaries = all_results[player].get("Boundaries (4s + 6s)", 0)
                plt.bar(player, boundaries, color=player_colors.get(player, 'gray'))
        plt.title("Boundaries")
        plt.xlabel("Players")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # graph peak year
    if "Peak Year" in selected_metrics:
        peak_year_data = {}
        for player, metrics in all_results.items():
            peak = metrics.get("Peak Year")
            if peak and isinstance(peak, str) and "Runs:" in peak:
                year = int(peak.split()[0])
                runs = float(peak.split("Runs:")[1].split(',')[0].strip())
                peak_year_data[player] = (year, runs)

        # plot bar chart
        if peak_year_data:
            players = list(peak_year_data.keys())
            years = [peak_year_data[p][0] for p in players]
            runs = [peak_year_data[p][1] for p in players]

            plt.figure(figsize=(10, 6))
            colors = [player_colors.get(player, 'gray') for player in players]  # Default to gray if no color is defined
            bars = plt.bar(players, runs, color=colors)
            plt.title("Peak Year (Runs) per Player")
            plt.xlabel("Players")
            plt.ylabel("Runs")
            plt.ylim(0, max(runs) + 100)

            # annotate bars with year
            for bar, year in zip(bars, years):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + 10, f"Year: {year}", ha='center')

            plt.tight_layout()
            plt.show()

    # line graph for year range stats
    if '2' in selected_filters:
        plt.figure(figsize=(10, 6))
        for player in all_results:
            df = pd.read_csv(files[player])
            df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
            yearly_runs = df.groupby('Year')['Runs'].sum()
            plt.plot(yearly_runs.index, yearly_runs.values, label=player, color=player_colors.get(player, 'gray'), linewidth=2)
        plt.title("Performance Over Years")
        plt.xlabel("Year")
        plt.ylabel("Total Runs")
        plt.legend()
        plt.tight_layout()
        plt.show()

# ask if want to plot graphs
display_graphs = get_yes_no("Do You Want to Display Graphs for the Selected Metrics? (yes/no): ")

if display_graphs == 'yes':
    plot_metrics()