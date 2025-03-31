# NCAA March Madness Predictor

A comprehensive machine learning system for predicting NCAA basketball tournament outcomes using historical data and advanced analytics.

## Project Overview

This project creates a prediction engine for the NCAA March Madness basketball tournament, leveraging historical tournament results and team performance metrics from KenPom.com. The system scrapes data, engineers features based on team matchups, trains multiple predictive models, and simulates tournament outcomes with different prediction strategies.

### Key Features

- **Data Collection Pipeline**: Scrapes historical NCAA tournament brackets and team statistics from KenPom.com 
- **Feature Engineering**: Creates matchup-based features measuring team differentials in key statistical categories
- **Multiple Prediction Models**: Includes various models optimized for different prediction strategies:
  - "Chalk-Heavy" model (favors higher seeds)
  - "Upset-Heavy" model (more likely to predict upsets)
  - "Balanced" model (optimized for overall accuracy)
- **Tournament Simulator**: Simulates the entire tournament bracket, handling game dependencies
- **Visualization**: Generates bracket visualizations and upset predictions
- **Upset Analysis**: Identifies potential upsets by seed differential and win probability

## Data Sources

1. **Tournament Data**: Historical NCAA tournament results from 2011-2024, scraped from Sports-Reference.com
2. **Team Statistics**: Team performance metrics from KenPom.com, including:
   - Adjusted efficiency metrics (offense, defense)
   - Four Factors statistics
   - Tempo metrics
   - Strength of schedule
   - Team experience and height data

## Model Features

The system uses various statistical differentials between teams to predict outcomes:

- Adjusted efficiency margin differentials
- Offensive and defensive efficiency differentials
- Strength of schedule differentials
- Shooting percentage differentials
- Seed differentials
- Height and experience differentials
- Offensive rebounding differentials
- Cross-matchup metrics (Team 1's offense vs Team 2's defense)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/march-madness-predictor.git
cd march-madness-predictor

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/kenpom_data data/tournament_data output
```

## Usage

### 1. Data Collection

```bash
# Scrape NCAA tournament data
python src/data_collection/scrape_tourney_data.py

# Scrape KenPom data (requires KenPom.com subscription)
python src/data_collection/scrape_kenpom_data.py
```

### 2. Data Processing

```bash
# Clean and standardize team names
python src/data_processing/clean_team_names.py

# Create matchup features
python src/data_processing/feature_engineering.py
```

### 3. Model Training and Prediction

```bash
# Train models
python src/modeling/model_training.py

# Predict current tournament
python src/modeling/predict_bracket.py
```

## Results

The system generates three different bracket predictions with varying levels of risk:

1. **Chalk-Heavy Model**: Conservative predictions favoring higher seeds
2. **Balanced Model**: Optimal balance between favorites and reasonable upsets
3. **Upset-Heavy Model**: More aggressive predictions of upsets

Output includes:
- CSV files with full bracket predictions
- Visualizations of predicted brackets
- Analysis of potential upsets
- Round-by-round win probabilities

## Model Analysis

For a comprehensive analysis of model performance, feature importance, and upset patterns, see [MODEL_RESULTS.md](MODEL_RESULTS.md).

This detailed report includes:
- Performance metrics for each model variant
- Prediction accuracy by tournament round
- Feature importance analysis
- Upset frequency patterns
- Key insights about tournament dynamics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KenPom.com for the advanced team metrics
- Sports-Reference.com for historical tournament data
