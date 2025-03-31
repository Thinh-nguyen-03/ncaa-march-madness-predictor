import os
import pandas as pd

# Create the mapping dictionary
final_name_mapping = {
    "Brigham Young": "BYU",
    "Florida State": "Florida St.",
    "Indiana State": "Indiana St.",
    "Kansas State": "Kansas St.",
    "LIU": "LIU Brooklyn",
    "Michigan State": "Michigan St.",
    "Morehead State": "Morehead St.",
    "Ohio State": "Ohio St.",
    "Penn State": "Penn St.",
    "Pitt": "Pittsburgh",
    "San Diego State": "San Diego St.",
    "St. John's (NY)": "St. John's",
    "Utah State": "Utah St.",
    "Virginia Commonwealth": "VCU",
    "Colorado State": "Colorado St.",
    "Detroit Mercy": "Detroit Mercy",
    "Iowa State": "Iowa St.",
    "Long Beach State": "Long Beach St.",
    "Loyola (MD)": "Loyola (MD)",
    "Murray State": "Murray St.",
    "New Mexico State": "New Mexico St.",
    "Norfolk State": "Norfolk St.",
    "North Carolina State": "NC State",
    "South Dakota State": "South Dakota St.",
    "Wichita State": "Wichita St.",
    "Albany (NY)": "Albany",
    "Miami (FL)": "Miami (FL)",
    "Northwestern State": "Northwestern St.",
    "Oklahoma State": "Oklahoma St.",
    "Arizona State": "Arizona St.",
    "Louisiana": "LSU",
    "Louisiana State": "LSU",
    "North Dakota State": "North Dakota St.",
    "UMass": "UMass",
    "Georgia State": "Georgia St.",
    "Cal State Bakersfield": "Cal St. Bakersfield",
    "Fresno State": "Fresno St.",
    "Little Rock": "Little Rock",
    "Oregon State": "Oregon St.",
    "Southern California": "USC",
    "Weber State": "Weber St.",
    "ETSU": "ETSU",
    "Jacksonville State": "Jacksonville St.",
    "Kent State": "Kent St.",
    "UC-Davis": "UC Davis",
    "Cal State Fullerton": "Cal St. Fullerton",
    "Loyola (IL)": "Loyola Chicago",
    "Texas Christian": "TCU",
    "Wright State": "Wright St.",
    "Central Florida": "UCF",
    "Gardner-Webb": "Gardner-Webb",
    "Mississippi State": "Mississippi",
    "Cleveland State": "Cleveland St.",
    "Boise State": "Boise St.",
    "Montana State": "Montana St.",
    "College of Charleston": "Charleston",
    "Kennesaw State": "Kennesaw St.",
    "Texas A&M-Corpus Christi": "Texas A&M-CC",
    "Grambling State": "Grambling",
    "McNeese State": "McNeese St.",
    "Washington State": "Washington St.",
    "UC Santa Barbara": "UCSB"
}

# Load tournament results and build a per-year team set
tourney_file = r"data\tournament_data\ncaa_tournament_results_simplified.csv"
kenpom_dir = r"data\kenpom_data"
tourney_df = pd.read_csv(tourney_file)

# Build a dictionary: year -> set of tournament teams (from team1 and team2)
tourney_teams_by_year = {}
for year in tourney_df['year'].unique():
    try:
        yr = int(year)
    except ValueError:
        continue
    # Skip 2020 (COVID year)
    if yr == 2020:
        continue
    year_df = tourney_df[tourney_df['year'] == yr]
    # Get unique teams from both columns
    teams = set(year_df['team1'].dropna().unique()).union(set(year_df['team2'].dropna().unique()))
    # Standardize names using mapping
    standardized_teams = {final_name_mapping.get(team, team) for team in teams}
    tourney_teams_by_year[yr] = standardized_teams

# Process KenPom merged files for each year, filter to only tournament teams, and keep only desired columns
columns_to_keep = [
    "Year",
    "Rk",
    "TeamName",
    "Seed",
    "AdjEM",
    "AdjO",
    "AdjD",
    "AdjT",
    "SOS-AdjEM",
    "SOS-OppO",
    "SOS-OppD",
    "NCSOS-AdjEM",
    "Tempo-Adj",
    "Avg. Poss Length-Offense",
    "Avg. Poss Length-Defense",
    "Off. Efficiency-Adj",
    "Off. Efficiency-Raw",
    "Def. Efficiency-Adj",
    "Def. Efficiency-Raw",
    "Off-eFG%",
    "Off-TO%",
    "Off-OR%",
    "Off-FTRate",
    "Def-eFG%",
    "Def-TO%",
    "Def-OR%",
    "Def-FTRate",
    "AvgHgt",
    "Experience",
    "Bench",
    "Continuity",
    "Off-FT",
    "Off-2P",
    "Off-3P",
    "Def-FT",
    "Def-2P",
    "Def-3P"
]

start_year = 2011
end_year = 2025

cleaned_dfs = []

for year in range(start_year, end_year + 1):
    if year == 2020:
        print(f"Skipping {year} as it is marked to skip.")
        continue

    kenpom_file = os.path.join(kenpom_dir, f"kenpom_merged_{year}.csv")
    if not os.path.exists(kenpom_file):
        print(f"KenPom merged file for {year} not found: {kenpom_file}. Skipping.")
        continue

    try:
        kenpom_df = pd.read_csv(kenpom_file)
    except Exception as e:
        print(f"Error reading {kenpom_file}: {e}")
        continue

    # Make sure the team name column is named 'TeamName'
    if 'TeamName' not in kenpom_df.columns and 'Team' in kenpom_df.columns:
        kenpom_df.rename(columns={'Team': 'TeamName'}, inplace=True)
    if 'TeamName' not in kenpom_df.columns:
        print(f"'TeamName' column not found in {kenpom_file}. Skipping.")
        continue

    # Optionally add a Year column if missing
    if 'Year' not in kenpom_df.columns:
        kenpom_df['Year'] = year

    # For consistency, you might also strip extra whitespace from team names
    kenpom_df['TeamName'] = kenpom_df['TeamName'].str.strip()

    # Filter the kenpom data: only keep rows where the TeamName is in the standardized tournament teams for that year
    tournament_teams = tourney_teams_by_year.get(year, set())
    filtered_df = kenpom_df[kenpom_df['TeamName'].isin(tournament_teams)].copy()

    print(f"For {year}, kept {len(filtered_df)} teams out of {len(kenpom_df)} in the KenPom file.")

    # Keep only the desired columns
    filtered_df = filtered_df[[col for col in columns_to_keep if col in filtered_df.columns]]

    cleaned_dfs.append(filtered_df)

# Merge all years into one big file and save
if cleaned_dfs:
    merged_all = pd.concat(cleaned_dfs, ignore_index=True)
    output_file = os.path.join(kenpom_dir, "kenpom_all_tournament_teams_cleaned.csv")
    merged_all.to_csv(output_file, index=False)
    print(f"Saved merged KenPom tournament team data to: {output_file}")
else:
    print("No cleaned KenPom data available to merge.")
