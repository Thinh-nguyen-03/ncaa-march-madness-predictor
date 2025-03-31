import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

kenpom_data = pd.read_csv(r"data\kenpom_data\kenpom_all_tournament_teams_cleaned.csv")
tourney_results = pd.read_csv(r"data\tournament_data\ncaa_tournament_results_simplified.csv")

# Check for data quality and consistency
print(f"KenPom data shape: {kenpom_data.shape}")
print(f"Tournament results shape: {tourney_results.shape}")

# Examine team name consistency
kenpom_teams = set(kenpom_data['TeamName'])
tourney_teams = set(tourney_results['team1']).union(set(tourney_results['team2']))
missing_teams = tourney_teams - kenpom_teams

if missing_teams:
    print(f"Teams in tournament data but missing from KenPom: {missing_teams}")
    # Create a mapping dictionary to standardize these names

# Create a function to build matchup features
def create_matchup_features(tourney_df, kenpom_df):
    matchups = []

    # Define the stats to use
    exclude_columns = ['Year', 'Rk', 'TeamName', 'Seed']
    all_stats = [col for col in kenpom_df.columns if col not in exclude_columns]

    # Loop through all tournament games
    for _, game in tourney_df.iterrows():
        year = game['year']
        team1 = game['team1']
        team2 = game['team2']
        winner = game['winner']

        # Get KenPom stats for both teams in that year
        team1_stats = kenpom_df[(kenpom_df['Year'] == year) & (kenpom_df['TeamName'] == team1)]
        team2_stats = kenpom_df[(kenpom_df['Year'] == year) & (kenpom_df['TeamName'] == team2)]

        # Skip if stats not found for either team
        if team1_stats.empty or team2_stats.empty:
            print(f"Missing KenPom data: Year {year}, {team1} vs {team2}")
            continue

        # Create feature dictionary
        features = {}
        features['year'] = year
        features['round'] = game['round']
        features['team1'] = team1
        features['team2'] = team2

        # Add seed information
        features['seed1'] = game['team1_seed']
        features['seed2'] = game['team2_seed']
        features['seed_diff'] = game['team1_seed'] - game['team2_seed']

        # Add raw and differential features for every KenPom stat
        for stat in all_stats:
            try:
                stat1 = float(team1_stats[stat].iloc[0])
                stat2 = float(team2_stats[stat].iloc[0])

                # Store raw values
                features[f'{stat}_1'] = stat1
                features[f'{stat}_2'] = stat2

                # Store differential (team1 - team2)
                features[f'{stat}_diff'] = stat1 - stat2

                # Special case for defensive metrics (lower is better)
                if stat in ['AdjD', 'Def. Efficiency-Adj', 'Def. Efficiency-Raw',
                           'Def-eFG%', 'Def-OR%', 'Def-FTRate', 'Def-FT', 'Def-2P', 'Def-3P']:
                    features[f'{stat}_diff_inv'] = -features[f'{stat}_diff']

            except (ValueError, TypeError):
                # Skip if stat cannot be converted to float
                print(f"Error processing stat {stat} for {team1} vs {team2} in {year}")
                continue

        # Additional compound features

        # Offensive vs Defensive matchup (team1 offense against team2 defense)
        if 'AdjO_1' in features and 'AdjD_2' in features:
            features['Off1_vs_Def2'] = features['AdjO_1'] - features['AdjD_2']

        # Defensive vs Offensive matchup (team1 defense against team2 offense)
        if 'AdjD_1' in features and 'AdjO_2' in features:
            features['Def1_vs_Off2'] = features['AdjD_1'] - features['AdjO_2']

        # Overall net matchup advantage
        if 'Off1_vs_Def2' in features and 'Def1_vs_Off2' in features:
            features['Net_Matchup_Advantage'] = features['Off1_vs_Def2'] - features['Def1_vs_Off2']

        # Experience advantage weighted by continuity
        if all(k in features for k in ['Experience_1', 'Experience_2', 'Continuity_1', 'Continuity_2']):
            features['Weighted_Exp_Advantage'] = (features['Experience_1'] * features['Continuity_1']) - (
                features['Experience_2'] * features['Continuity_2'])

        # Target variable - 1 if team1 won, 0 if team2 won
        features['team1_won'] = 1 if winner == team1 else 0

        matchups.append(features)

    return pd.DataFrame(matchups)

# Create the feature dataset
matchup_features = create_matchup_features(tourney_results, kenpom_data)

# Save processed data
matchup_features.to_csv('data\matchup_features_2011_2024.csv', index=False)

# Feature exploration and analysis
print(f"\nCreated feature dataset with {len(matchup_features)} matchups and {matchup_features.columns.size} columns")

# Check for any missing values
missing_values = matchup_features.isnull().sum()
print(f"\nColumns with missing values:")
print(missing_values[missing_values > 0])

# Analyze correlation with winning
feature_cols = [col for col in matchup_features.columns if (col.endswith('_diff') or col.endswith('_diff_inv')
                or col in ['Net_Matchup_Advantage', 'Weighted_Exp_Advantage', 'seed_diff'])
                and col != 'team1_won']

# Filter out features with missing values for correlation analysis
complete_features = [col for col in feature_cols if matchup_features[col].isnull().sum() == 0]
correlation = matchup_features[complete_features + ['team1_won']].corr()['team1_won'].drop('team1_won').sort_values(ascending=False)

print("\nTop 15 features by correlation with winning:")
print(correlation.head(15))
print("\nBottom 15 features by correlation with winning:")
print(correlation.tail(15))

# Create a visualization of the top features
plt.figure(figsize=(12, 10))
top_features = correlation.head(15).index
sns.barplot(x=correlation.head(15).values, y=top_features, palette='coolwarm')
plt.title('Top 15 Features by Correlation with Winning', fontsize=14)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.tight_layout()
plt.savefig(r'output\top_features_correlation.png')

# Additional analysis: win rate by seed difference
matchup_features['seed_gap'] = abs(matchup_features['seed1'] - matchup_features['seed2'])
matchup_features['higher_seed_won'] = np.where(
    ((matchup_features['seed1'] < matchup_features['seed2']) & (matchup_features['team1_won'] == 1)) |
    ((matchup_features['seed2'] < matchup_features['seed1']) & (matchup_features['team1_won'] == 0)),
    1, 0
)

seed_performance = matchup_features.groupby('seed_gap').agg(
    total_games=('team1_won', 'count'),
    higher_seed_wins=('higher_seed_won', 'sum')
).reset_index()

seed_performance['win_rate'] = seed_performance['higher_seed_wins'] / seed_performance['total_games']

plt.figure(figsize=(10, 6))
sns.barplot(x='seed_gap', y='win_rate', data=seed_performance)
plt.title('Higher Seed Win Rate by Seed Difference', fontsize=14)
plt.xlabel('Seed Difference', fontsize=12)
plt.ylabel('Higher Seed Win Rate', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'output\seed_difference_win_rate.png')

# Calculate upset frequency by seed matchup
plt.figure(figsize=(12, 8))
pivot = pd.pivot_table(
    matchup_features,
    values='higher_seed_won',
    index='seed1',
    columns='seed2',
    aggfunc=lambda x: 1 - np.mean(x)  # Showing upset frequency (1 - win rate)
)
sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Upset Frequency by Seed Matchup', fontsize=14)
plt.xlabel('Team 2 Seed', fontsize=12)
plt.ylabel('Team 1 Seed', fontsize=12)
plt.tight_layout()
plt.savefig(r'output\upset_frequency_heatmap.png')

print("\nFeature engineering complete! Ready for model building.")