import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import io
from pathlib import Path

# Set style for visualizations
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

print("Loading and preparing data...")

# Load the processed matchup features for training
matchup_features = pd.read_csv(r'data\matchup_features_2011_2024.csv')
print(f"Loaded matchup features dataset with {len(matchup_features)} historical matchups")

# Load KenPom data for teams
kenpom_data = pd.read_csv(r'data\kenpom_data\kenpom_all_tournament_teams_cleaned.csv')
kenpom_2025 = kenpom_data[kenpom_data['Year'] == 2025]
if len(kenpom_2025) == 0:
    print("WARNING: No KenPom data for 2025 available! Using 2024 data as placeholder.")
    kenpom_2025 = kenpom_data[kenpom_data['Year'] == 2024]

print(f"Found KenPom data for {len(kenpom_2025)} teams from year {kenpom_2025['Year'].iloc[0]}")

# Prepare features and target variable for model training
important_features = [
    'AdjEM_diff', 'AdjO_diff', 'SOS-AdjEM_diff', 'Off. Efficiency-Adj_diff',
    'AdjD_diff_inv', 'SOS-OppO_diff', 'Off-eFG%_diff', 'Def-eFG%_diff_inv',
    'seed_diff', 'AvgHgt_diff', 'Experience_diff', 'Off-OR%_diff',
    'Off1_vs_Def2', 'Def1_vs_Off2'
]

# Make sure all selected features exist in the dataset
available_features = [col for col in important_features if col in matchup_features.columns]
print(f"Using {len(available_features)} features: {available_features}")

# Analyze relationship between seed_diff and winning for debugging
if 'seed_diff' in matchup_features.columns:
    won_games = matchup_features[matchup_features['team1_won'] == 1]
    lost_games = matchup_features[matchup_features['team1_won'] == 0]
    print(f"Mean seed_diff when team1 won: {won_games['seed_diff'].mean()}")
    print(f"Mean seed_diff when team1 lost: {lost_games['seed_diff'].mean()}")
    print(f"Correlation between seed_diff and team1_won: {matchup_features[['seed_diff', 'team1_won']].corr().iloc[0,1]:.4f}")

train_mask = matchup_features['year'] <= 2024  # All dat

# Create 2011-2024 training set
X_train = matchup_features[train_mask][available_features]
y_train = matchup_features[train_mask]['team1_won']
print(f"Training Set: 2011-2024 with {len(X_train)} matchups")

# Scale features for the training set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Train different models with different parameters
model_params = {
    'Upset-Heavy': {'n_estimators': 200, 'learning_rate': 0.2, 'max_depth': 5, 'random_state': 42},
    'Chalk-Heavy': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3, 'random_state': 42},
    'Balanced': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 4, 'random_state': 42},
}

# Train 2011-2024 models
models = {}
for name, params in model_params.items():
    print(f"Training {name} model on 2011-2024 data...")
    model = GradientBoostingClassifier(**params)
    model.fit(X_scaled, y_train)
    models[name] = model

print("All models trained!")

# Load 2025 tournament bracket data from CSV
tournament_data = pd.read_csv(r'data\tournament_data\ncaa_tournament_results_simplified.csv')
print(f"Loaded tournament data with {len(tournament_data)} games")

# Filter to get only 2025 tournament with TBD winners
bracket_2025 = tournament_data[(tournament_data['year'] == 2025) & (tournament_data['winner'] == 'TBD')]

if len(bracket_2025) == 0:
    print("ERROR: No 2025 tournament games with TBD winners found in the CSV!")
else:
    print(f"Found {len(bracket_2025)} games for 2025 tournament with TBD winners")

# Assign regions to the games (every 8 first-round games form a region)
first_round = bracket_2025[bracket_2025['round'] == 1].copy()
regions = ['East', 'Midwest', 'South', 'West']
region_assignments = []

for i in range(len(first_round)):
    region_idx = i // 8
    if region_idx < len(regions):
        region_assignments.append(regions[region_idx])
    else:
        region_assignments.append('Unknown')

first_round['region'] = region_assignments

# Create a region mapping for later use
team_region_map = {}
for _, row in first_round.iterrows():
    team_region_map[row['team1']] = row['region']
    team_region_map[row['team2']] = row['region']

print(f"Assigned regions to first round games: {dict(zip(regions, [sum(first_round['region'] == r) for r in regions]))}")

# Function to predict a single matchup
def predict_matchup(team1, team2, seed1, seed2, model, scaler, features):
    """Predict the winner of a matchup between team1 and team2"""
    # Get team stats from KenPom data
    team1_stats = kenpom_2025[kenpom_2025['TeamName'] == team1]
    team2_stats = kenpom_2025[kenpom_2025['TeamName'] == team2]

    # Check if either team is missing from KenPom data
    if team1_stats.empty:
        print(f"WARNING: {team1} not found in KenPom data! Using seed-based prediction.")
        predicted_winner = team1 if seed1 < seed2 else team2
        predicted_winner_seed = seed1 if predicted_winner == team1 else seed2
        win_prob = 0.75 if seed1 < seed2 else 0.25

        return {
            'team1': team1,
            'team2': team2,
            'team1_seed': seed1,
            'team2_seed': seed2,
            'predicted_winner': predicted_winner,
            'predicted_winner_seed': predicted_winner_seed,
            'win_probability': win_prob,
            'team1_win_probability': win_prob if predicted_winner == team1 else 1-win_prob,
            'team2_win_probability': win_prob if predicted_winner == team2 else 1-win_prob,
            'predicted_by': 'seed'
        }

    if team2_stats.empty:
        print(f"WARNING: {team2} not found in KenPom data! Using seed-based prediction.")
        predicted_winner = team1 if seed1 < seed2 else team2
        predicted_winner_seed = seed1 if predicted_winner == team1 else seed2
        win_prob = 0.75 if seed1 < seed2 else 0.25

        return {
            'team1': team1,
            'team2': team2,
            'team1_seed': seed1,
            'team2_seed': seed2,
            'predicted_winner': predicted_winner,
            'predicted_winner_seed': predicted_winner_seed,
            'win_probability': win_prob,
            'team1_win_probability': win_prob if predicted_winner == team1 else 1-win_prob,
            'team2_win_probability': win_prob if predicted_winner == team2 else 1-win_prob,
            'predicted_by': 'seed'
        }

    # Get first row for each team stats
    team1_stats = team1_stats.iloc[0]
    team2_stats = team2_stats.iloc[0]

    # Create feature vector
    matchup_features = {}

    # Add seed information
    matchup_features['seed_diff'] = seed1 - seed2

    # Process each model feature
    for feature in features:
        if feature == 'seed_diff':
            continue  # Already added

        # Extract the base statistic name
        if '_diff' in feature:
            base_stat = feature.replace('_diff', '')
            if base_stat.endswith('_inv'):
                base_stat = base_stat.replace('_inv', '')
        else:
            base_stat = feature

        if base_stat == 'Off1_vs_Def2':
            matchup_features[feature] = float(team1_stats['AdjO']) - float(team2_stats['AdjD'])
            continue

        if base_stat == 'Def1_vs_Off2':
            matchup_features[feature] = float(team1_stats['AdjD']) - float(team2_stats['AdjO'])
            continue

        # Regular differential features
        if base_stat in team1_stats and base_stat in team2_stats:
            try:
                stat1 = float(team1_stats[base_stat])
                stat2 = float(team2_stats[base_stat])

                if '_inv' in feature:
                    # Inverse differential (for defensive stats)
                    matchup_features[feature] = -(stat1 - stat2)
                else:
                    # Regular differential
                    matchup_features[feature] = stat1 - stat2
            except (ValueError, TypeError):
                # If stats cannot be converted to float
                matchup_features[feature] = 0
                print(f"Warning: Could not process {base_stat} for {team1} vs {team2}")

    # Create DataFrame with the same structure as training data
    X_pred = pd.DataFrame([matchup_features])

    # Fill missing features with 0 (if any)
    for feature in features:
        if feature not in X_pred.columns:
            X_pred[feature] = 0

    # Ensure columns are in the same order as training data
    X_pred = X_pred[features]

    # Scale features
    X_pred_scaled = scaler.transform(X_pred)

    # Predict
    win_prob = model.predict_proba(X_pred_scaled)[0][1]
    predicted_winner = team1 if win_prob > 0.5 else team2
    predicted_winner_seed = seed1 if predicted_winner == team1 else seed2

    result = {
        'team1': team1,
        'team2': team2,
        'team1_seed': seed1,
        'team2_seed': seed2,
        'team1_win_probability': win_prob,
        'team2_win_probability': 1 - win_prob,
        'predicted_winner': predicted_winner,
        'predicted_winner_seed': predicted_winner_seed,
        'win_probability': max(win_prob, 1-win_prob),
        'predicted_by': 'model'
    }

    return result

# Function to simulate the tournament with a specified model
def simulate_tournament(tournament_df, model, scaler, features, model_name="default", data_years="2011-2024", save_to_csv=True):
    """
    Simulate the entire NCAA tournament, filling in winners for each game
    and generating next-round matchups dynamically.
    Updates the original DataFrame and optionally saves back to CSV.
    """
    print(f"\nSimulating 2025 NCAA Tournament with {model_name} model trained on {data_years} data...")

    # Create a working copy to avoid modifying the original until we're done
    working_df = tournament_df.copy()
    if 'predicted_probability' not in working_df.columns:
        working_df['predicted_probability'] = np.nan

    # Get only 2025 first round games
    first_round = working_df[(working_df['year'] == 2025) & (working_df['round'] == 1) & (working_df['winner'] == 'TBD')].copy()

    # Create a new DataFrame to hold all tournament games including future rounds
    tournament_games = first_round.copy()

    # Dictionary to track winners for team lookups in later rounds
    winners = {}

    # Process each round sequentially
    max_round = 6  # Championship game is round 6
    current_round = 1

    while current_round <= max_round:
        print(f"\nRound {current_round} predictions [{model_name} - {data_years}]:")
        round_games = tournament_games[tournament_games['round'] == current_round].copy()

        # Skip if no games in this round (shouldn't happen with proper generation)
        if len(round_games) == 0:
            print(f"  No games found for round {current_round}")
            break

        # Predict all games in this round
        for idx, game in round_games.iterrows():
            team1, team2 = game['team1'], game['team2']
            seed1, seed2 = game['team1_seed'], game['team2_seed']

            # Find regions for display
            region1 = team_region_map.get(team1, "Unknown")
            region2 = team_region_map.get(team2, "Unknown")

            # Predict the matchup
            result = predict_matchup(team1, team2, seed1, seed2, model, scaler, features)

            # Update the tournament games DataFrame with prediction
            winner = result['predicted_winner']
            winner_seed = result['predicted_winner_seed']
            win_prob = result['win_probability']

            # Store the result in our tournament games DataFrame
            tournament_games.at[idx, 'winner'] = winner
            tournament_games.at[idx, 'predicted_probability'] = win_prob

            # Store winner for reference in later rounds
            game_id = idx if isinstance(idx, str) else str(idx)
            winners[game_id] = {'team': winner, 'seed': winner_seed, 'region': region1 if winner == team1 else region2}

            # Print prediction with formatting based on round
            if current_round <= 4:  # Rounds within regions
                region = region1 if region1 == region2 else f"{region1}/{region2}"
                print(f"  {region}: ({seed1}) {team1} vs ({seed2}) {team2} → {winner} ({win_prob:.2f})")
            elif current_round == 5:  # Final Four
                print(f"  Final Four: ({seed1}) {team1} ({region1}) vs ({seed2}) {team2} ({region2}) → {winner} ({win_prob:.2f})")
            else:  # Championship
                print(f"  Championship: ({seed1}) {team1} ({region1}) vs ({seed2}) {team2} ({region2}) → {winner} ({win_prob:.2f})")

        # If not at the final round, generate next round matchups
        if current_round < max_round:
            next_round = current_round + 1

            # Generate next round matchups - different logic for different rounds
            next_round_games = []

            if current_round == 1:  # First round -> Second round (32 -> 16 teams)
                # In the second round, winners of consecutive games play each other
                for i in range(0, len(round_games), 2):
                    if i + 1 < len(round_games):
                        game1_idx = round_games.index[i]
                        game2_idx = round_games.index[i + 1]

                        winner1 = winners[str(game1_idx)]
                        winner2 = winners[str(game2_idx)]

                        # Determine region based on the games
                        region = winner1['region'] # Should be the same region

                        # Create the next round matchup
                        next_game = {
                            'year': 2025,
                            'round': next_round,
                            'team1': winner1['team'],
                            'team2': winner2['team'],
                            'team1_seed': winner1['seed'],
                            'team2_seed': winner2['seed'],
                            'winner': 'TBD',
                            'predicted_probability': np.nan
                        }

                        next_round_games.append(next_game)

            elif current_round == 2:  # Second round -> Sweet 16 (16 -> 8 teams)
                # Same pattern as round 1 -> round 2
                for i in range(0, len(round_games), 2):
                    if i + 1 < len(round_games):
                        game1_idx = round_games.index[i]
                        game2_idx = round_games.index[i + 1]

                        winner1 = winners[str(game1_idx)]
                        winner2 = winners[str(game2_idx)]

                        # Create the next round matchup
                        next_game = {
                            'year': 2025,
                            'round': next_round,
                            'team1': winner1['team'],
                            'team2': winner2['team'],
                            'team1_seed': winner1['seed'],
                            'team2_seed': winner2['seed'],
                            'winner': 'TBD',
                            'predicted_probability': np.nan
                        }

                        next_round_games.append(next_game)

            elif current_round == 3:  # Sweet 16 -> Elite 8 (8 -> 4 teams)
                # Same pattern again
                for i in range(0, len(round_games), 2):
                    if i + 1 < len(round_games):
                        game1_idx = round_games.index[i]
                        game2_idx = round_games.index[i + 1]

                        winner1 = winners[str(game1_idx)]
                        winner2 = winners[str(game2_idx)]

                        # Create the next round matchup
                        next_game = {
                            'year': 2025,
                            'round': next_round,
                            'team1': winner1['team'],
                            'team2': winner2['team'],
                            'team1_seed': winner1['seed'],
                            'team2_seed': winner2['seed'],
                            'winner': 'TBD',
                            'predicted_probability': np.nan
                        }

                        next_round_games.append(next_game)

            elif current_round == 4:  # Elite 8 -> Final Four (4 -> 2 teams)
                # Special pairing for Final Four: South vs West, East vs Midwest
                region_winners = {}

                # Collect the regional winners
                for idx, row in round_games.iterrows():
                    winner_info = winners[str(idx)]
                    region_winners[winner_info['region']] = winner_info

                # Create Final Four matchups
                if 'South' in region_winners and 'West' in region_winners:
                    next_game = {
                        'year': 2025,
                        'round': next_round,
                        'team1': region_winners['South']['team'],
                        'team2': region_winners['West']['team'],
                        'team1_seed': region_winners['South']['seed'],
                        'team2_seed': region_winners['West']['seed'],
                        'winner': 'TBD',
                        'predicted_probability': np.nan
                    }
                    next_round_games.append(next_game)
                else:
                    print("WARNING: Missing region winners for South vs West matchup")

                if 'East' in region_winners and 'Midwest' in region_winners:
                    next_game = {
                        'year': 2025,
                        'round': next_round,
                        'team1': region_winners['East']['team'],
                        'team2': region_winners['Midwest']['team'],
                        'team1_seed': region_winners['East']['seed'],
                        'team2_seed': region_winners['Midwest']['seed'],
                        'winner': 'TBD',
                        'predicted_probability': np.nan
                    }
                    next_round_games.append(next_game)
                else:
                    print("WARNING: Missing region winners for East vs Midwest matchup")

            elif current_round == 5:  # Final Four -> Championship (2 -> 1 team)
                if len(round_games) == 2:
                    game1_idx = round_games.index[0]
                    game2_idx = round_games.index[1]

                    winner1 = winners[str(game1_idx)]
                    winner2 = winners[str(game2_idx)]

                    # Create championship game
                    next_game = {
                        'year': 2025,
                        'round': next_round,
                        'team1': winner1['team'],
                        'team2': winner2['team'],
                        'team1_seed': winner1['seed'],
                        'team2_seed': winner2['seed'],
                        'winner': 'TBD',
                        'predicted_probability': np.nan
                    }

                    next_round_games.append(next_game)

            # Add the new games to our tournament games DataFrame
            if next_round_games:
                next_round_df = pd.DataFrame(next_round_games)
                tournament_games = pd.concat([tournament_games, next_round_df], ignore_index=True)

        # Move to next round
        current_round += 1

    # Get the champion
    championship_games = tournament_games[tournament_games['round'] == max_round]
    if not championship_games.empty and 'winner' in championship_games.columns:
        champion = championship_games['winner'].iloc[0]
        print(f"\nPredicted Champion [{model_name} - {data_years}]: {champion}")
    else:
        champion = "Unknown"
        print(f"\nCould not determine champion for {model_name} model trained on {data_years}")

    # Save the updated DataFrame back to CSV
    if save_to_csv:
        # Merge our tournament games with the original data
        all_games = pd.concat([
            # Non-2025 games from original data
            working_df[working_df['year'] != 2025],
            # 2025 games that aren't round 1 TBD games
            working_df[(working_df['year'] == 2025) & ~((working_df['round'] == 1) & (working_df['winner'] == 'TBD'))],
            # Our simulated tournament games
            tournament_games
        ], ignore_index=True)

        # Save to a new CSV
        output_filename = f'output\\ncaa_tournament_results_{model_name.lower()}_{data_years}_predictions.csv'
        all_games.to_csv(output_filename, index=False)
        print(f"Updated tournament data saved to '{output_filename}'")

    return tournament_games, champion

# Identify potential upsets in predictions
def identify_upsets(predicted_df, model_name="default", data_years="2011-2024"):
    """Identify potential upsets in the predicted bracket"""
    # Filter to tournament games
    tournament_games = predicted_df[predicted_df['winner'] != 'TBD']

    upsets = []

    for _, game in tournament_games.iterrows():
        team1_seed = game['team1_seed']
        team2_seed = game['team2_seed']
        winner = game['winner']

        # Check if lower seed is predicted to win
        is_upset = (team1_seed > team2_seed and winner == game['team1']) or \
                   (team2_seed > team1_seed and winner == game['team2'])

        upset_margin = abs(team1_seed - team2_seed)

        if is_upset and upset_margin >= 2:  # Only count significant upsets
            upset_info = {
                'round': game['round'],
                'favorite': game['team2'] if winner == game['team1'] else game['team1'],
                'favorite_seed': team2_seed if winner == game['team1'] else team1_seed,
                'underdog': winner,
                'underdog_seed': team1_seed if winner == game['team1'] else team2_seed,
                'win_probability': game['predicted_probability'],
                'upset_margin': upset_margin
            }

            upsets.append(upset_info)

    # Sort upsets by round, then by upset margin
    upsets.sort(key=lambda x: (x['round'], -x['upset_margin']))

    # Print upset predictions
    round_names = {1: 'First Round', 2: 'Second Round', 3: 'Sweet 16',
                   4: 'Elite 8', 5: 'Final Four', 6: 'Championship'}

    print(f"\nPredicted Upsets [{model_name} - {data_years}]:")
    print("-" * 80)
    print(f"{'Round':<15} {'Matchup':<40} {'Prob':<10} {'Margin'}")
    print("-" * 80)

    for upset in upsets:
        round_name = round_names.get(upset['round'], f"Round {upset['round']}")
        matchup = f"({upset['underdog_seed']}) {upset['underdog']} over ({upset['favorite_seed']}) {upset['favorite']}"
        print(f"{round_name:<15} {matchup:<40} {upset['win_probability']:.2f}      {upset['upset_margin']}")

    return upsets

# Main execution function
def predict_march_madness():
    """Main function to run the complete March Madness prediction pipeline with all models"""
    # Ensure we have 2025 bracket data
    if len(bracket_2025) == 0:
        print("Error: No 2025 tournament data available with TBD winners.")
        return None

    results = {}

    # Run simulation with each model
    for model_name, model_instance in models.items():
        # Simulate tournament with this model
        predicted_bracket, champion = simulate_tournament(
            tournament_data,
            model_instance,
            scaler,
            available_features,
            model_name=model_name
        )

        # Identify upsets for this model
        upsets = identify_upsets(predicted_bracket, model_name=model_name)

        # Store results
        results[model_name] = {
            'bracket': predicted_bracket,
            'champion': champion,
            'upsets': upsets
        }

    # Compare results
    print("\n\n=============================================")
    print("COMPARISON OF MODEL PREDICTIONS")
    print("=============================================")
    print(f"{'Model':<15} {'Champion':<20} {'# of Upsets':<15}")
    print("---------------------------------------------")

    for model_name, result in results.items():
        num_upsets = len(result['upsets'])
        print(f"{model_name:<15} {result['champion']:<20} {num_upsets:<15}")

    return results

# Run the prediction pipeline
if __name__ == "__main__":
    results = predict_march_madness()