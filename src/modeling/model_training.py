import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
import pickle

# Load the processed matchup features
matchup_features = pd.read_csv(r'data\matchup_features_2011_2024.csv')
print(f"Loaded dataset with {len(matchup_features)} matchups")

# repare features and target variable
# Select the most important features based on correlation analysis
important_features = [
    'AdjEM_diff', 'AdjO_diff', 'SOS-AdjEM_diff', 'Off. Efficiency-Adj_diff',
    'AdjD_diff_inv', 'SOS-OppO_diff', 'Off-eFG%_diff', 'Def-eFG%_diff_inv',
    'seed_diff', 'AvgHgt_diff', 'Experience_diff', 'Off-OR%_diff',
    'Off1_vs_Def2', 'Def1_vs_Off2'
]

# Make sure all selected features exist in the dataset
available_features = [col for col in important_features if col in matchup_features.columns]
print(f"Using {len(available_features)} features: {available_features}")

# Split features and target
X = matchup_features[available_features]
y = matchup_features['team1_won']

# Temporal split - Train on older tournaments, test on recent ones
train_years = list(range(2011, 2024))  # Train on 2011-2022
test_years = [2024]              # Test on 2023-2024

# Create train/test masks
train_mask = matchup_features['year'].isin(train_years)
test_mask = matchup_features['year'].isin(test_years)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")

    # Fit model
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'log_loss': loss,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_prob
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Log Loss: {loss:.4f}")
    print(f"  AUC: {auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Model comparison visualization
metrics = ['Accuracy', 'AUC']
model_comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'AUC': [results[m]['auc'] for m in results]
})

model_comparison_long = pd.melt(model_comparison, id_vars=['Model'],
                               value_vars=metrics, var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=model_comparison_long)
plt.title('Model Performance Comparison', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.ylim(0.5, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'output\model_comparison.png')

# Select best model (based on AUC)
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name} with AUC: {results[best_model_name]['auc']:.4f}")

# Feature importance analysis (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances from Model', fontsize=14)
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(r'output\model_feature_importance.png')

    print("\nTop 10 features by importance:")
    for i in range(min(10, X.shape[1])):
        print(f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# Performance by tournament round
if 'round' in matchup_features.columns:
    round_performance = pd.DataFrame({
        'actual': y_test,
        'predicted': results[best_model_name]['predictions'],
        'probability': results[best_model_name]['probabilities'],
        'round': matchup_features[test_mask]['round']
    })

    accuracy_by_round = round_performance.groupby('round').apply(
        lambda x: accuracy_score(x['actual'], x['predicted'])
    ).reset_index()
    accuracy_by_round.columns = ['Round', 'Accuracy']

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Round', y='Accuracy', data=accuracy_by_round)
    plt.title('Prediction Accuracy by Tournament Round', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Tournament Round', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(r'output\accuracy_by_round.png')

    print("\nAccuracy by tournament round:")
    for _, row in accuracy_by_round.iterrows():
        print(f"Round {int(row['Round'])}: {row['Accuracy']:.4f}")

# Save the best model
with open(r'output\march_madness_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'features': available_features
    }, f)

print("\nBest model saved as 'march_madness_model.pkl'")

# Define functions for bracket prediction
def predict_matchup(team1, team2, seed1, seed2, stats_df, model, scaler, features):
    """Predict the winner of a matchup between team1 and team2"""
    # Get team stats
    team1_stats = stats_df[stats_df['TeamName'] == team1].iloc[0]
    team2_stats = stats_df[stats_df['TeamName'] == team2].iloc[0]

    # Create feature vector
    matchup_features = {}

    # Add seed information
    matchup_features['seed_diff'] = seed1 - seed2

    # Process each model feature
    for feature in features:
        if feature == 'seed_diff':
            continue

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
            stat1 = float(team1_stats[base_stat])
            stat2 = float(team2_stats[base_stat])

            if '_inv' in feature:
                # Inverse differential (for defensive stats)
                matchup_features[feature] = -(stat1 - stat2)
            else:
                # Regular differential
                matchup_features[feature] = stat1 - stat2

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

    result = {
        'team1': team1,
        'team2': team2,
        'team1_win_probability': win_prob,
        'team2_win_probability': 1 - win_prob,
        'predicted_winner': team1 if win_prob > 0.5 else team2,
        'predicted_winner_seed': seed1 if win_prob > 0.5 else seed2
    }

    return result

print("\nModel building complete! You now have a trained model ready for predicting games.")