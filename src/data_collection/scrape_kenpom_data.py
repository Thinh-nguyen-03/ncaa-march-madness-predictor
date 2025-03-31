import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
import argparse
from kenpompy.utils import login
from kenpompy import summary
from kenpompy import misc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"kenpom_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KenPomDataCollector:
    """
    Data collector for KenPom.com data focusing on tournament teams
    """

    def __init__(self, email, password, start_year=2011, end_year=2025,
                 tournament_data_path=r"data\tournament_data\ncaa_tournament_results_simplified.csv",
                 output_dir=r"data\kenpom_data"):
        self.email = email
        self.password = password
        self.start_year = start_year
        self.end_year = end_year
        self.tournament_data_path = tournament_data_path
        self.output_dir = output_dir
        self.browser = None
        self.tournament_teams = {}  # Store tournament teams by year
        self.kenpom_data = {}  # Store KenPom data by year

        # Team name mappings for standardization
        self.team_name_mapping = {
            'North Carolina': 'North Carolina',
            'UNC': 'North Carolina',
            'N Carolina': 'North Carolina',
            'N.C. State': 'North Carolina State',
            'NC State': 'North Carolina State',
            'North Carolina St.': 'North Carolina State',
            'N.C. St.': 'North Carolina State',
            'Virginia Tech': 'Virginia Tech',
            'VA Tech': 'Virginia Tech',
            'Florida State': 'Florida State',
            'Florida St.': 'Florida State',
            'Fla State': 'Florida State',
            'FSU': 'Florida State',
            'Miami (FL)': 'Miami (FL)',
            'Miami': 'Miami (FL)',
            'Miami, FL': 'Miami (FL)',
            'Georgia Tech': 'Georgia Tech',
            'Ga Tech': 'Georgia Tech',
            'Connecticut': 'Connecticut',
            'UConn': 'Connecticut',
            'UCONN': 'Connecticut',
            'Conn': 'Connecticut',
            'Villanova': 'Villanova',
            'Nova': 'Villanova',
            'St. John\'s': 'St. John\'s',
            'Saint John\'s': 'St. John\'s',
            'St John\'s': 'St. John\'s',
            'Michigan State': 'Michigan State',
            'Michigan St.': 'Michigan State',
            'Mich State': 'Michigan State',
            'Mich St.': 'Michigan State',
            'Mich. St.': 'Michigan State',
            'Ohio State': 'Ohio State',
            'Ohio St.': 'Ohio State',
            'OSU': 'Ohio State',
            'Wisconsin': 'Wisconsin',
            'Wisc': 'Wisconsin',
            'Wis.': 'Wisconsin',
            'Illinois': 'Illinois',
            'Ill.': 'Illinois',
            'Indiana': 'Indiana',
            'Ind.': 'Indiana',
            'Minnesota': 'Minnesota',
            'Minn': 'Minnesota',
            'Penn State': 'Penn State',
            'Penn St.': 'Penn State',
            'Kansas': 'Kansas',
            'Kan.': 'Kansas',
            'West Virginia': 'West Virginia',
            'W Virginia': 'West Virginia',
            'WVU': 'West Virginia',
            'Iowa State': 'Iowa State',
            'Iowa St.': 'Iowa State',
            'Oklahoma': 'Oklahoma',
            'Okla.': 'Oklahoma',
            'Oklahoma State': 'Oklahoma State',
            'Okla. State': 'Oklahoma State',
            'Oklahoma St.': 'Oklahoma State',
            'TCU': 'Texas Christian',
            'Texas Christian': 'Texas Christian',
            'Kansas State': 'Kansas State',
            'Kansas St.': 'Kansas State',
            'K-State': 'Kansas State',
            'Tennessee': 'Tennessee',
            'Tenn': 'Tennessee',
            'Florida': 'Florida',
            'Fla': 'Florida',
            'LSU': 'Louisiana State',
            'Louisiana State': 'Louisiana State',
            'La. State': 'Louisiana State',
            'Mississippi State': 'Mississippi State',
            'Miss State': 'Mississippi State',
            'Miss. State': 'Mississippi State',
            'Mississippi': 'Mississippi',
            'Ole Miss': 'Mississippi',
            'Miss': 'Mississippi',
            'Miss.': 'Mississippi',
            'South Carolina': 'South Carolina',
            'S Carolina': 'South Carolina',
            'Texas A&M': 'Texas A&M',
            'Arizona State': 'Arizona State',
            'Arizona St.': 'Arizona State',
            'USC': 'Southern California',
            'Southern California': 'Southern California',
            'S California': 'Southern California',
            'Oregon State': 'Oregon State',
            'Oregon St.': 'Oregon State',
            'Washington State': 'Washington State',
            'Washington St.': 'Washington State',
            'California': 'California',
            'Cal': 'California',
            'San Diego State': 'San Diego State',
            'SDSU': 'San Diego State',
            'Saint Mary\'s': 'Saint Mary\'s',
            'St. Mary\'s': 'Saint Mary\'s',
            'St Mary\'s': 'Saint Mary\'s',
            'BYU': 'Brigham Young',
            'Brigham Young': 'Brigham Young',
            'Wichita State': 'Wichita State',
            'Wichita St.': 'Wichita State',
            'VCU': 'Virginia Commonwealth',
            'Virginia Commonwealth': 'Virginia Commonwealth',
            'Saint Louis': 'Saint Louis',
            'St. Louis': 'Saint Louis',
            'St Louis': 'Saint Louis',
            'Loyola Chicago': 'Loyola Chicago',
            'Loyola-Chicago': 'Loyola Chicago',
            'New Mexico State': 'New Mexico State',
            'New Mexico St.': 'New Mexico State',
            'Central Florida': 'Central Florida',
            'UCF': 'Central Florida',
            'Cent. Florida': 'Central Florida',
            'Florida Gulf CoaSt.': 'Florida Gulf CoaSt.',
            'FGCU': 'Florida Gulf CoaSt.',
            'Fla Gulf CoaSt.': 'Florida Gulf CoaSt.',
            'FL Gulf CoaSt.': 'Florida Gulf CoaSt.',
            'FAU': 'Florida Atlantic',
            'Florida Atlantic': 'Florida Atlantic',
            'Saint Joseph\'s': 'Saint Joseph\'s',
            'St. Joseph\'s': 'Saint Joseph\'s',
            'St Joseph\'s': 'Saint Joseph\'s',
            'Saint Peter\'s': 'Saint Peter\'s',
            'St. Peter\'s': 'Saint Peter\'s',
            'St Peter\'s': 'Saint Peter\'s',
            'FDU': 'Fairleigh Dickinson',
            'Fairleigh Dickinson': 'Fairleigh Dickinson',
            'UNC Asheville': 'UNC Asheville',
            'Montana State': 'Montana State',
            'Montana St.': 'Montana State',
            'Jacksonville State': 'Jacksonville State',
            'Jacksonville St.': 'Jacksonville State',
            'South Dakota State': 'South Dakota State',
            'S Dakota State': 'South Dakota State',
            'North Dakota State': 'North Dakota State',
            'N Dakota State': 'North Dakota State',
            'UCSB': 'UC Santa Barbara',
            'UC Santa Barbara': 'UC Santa Barbara',
            'California-Santa Barbara': 'UC Santa Barbara',
            'Kent State': 'Kent State',
            'Kent St.': 'Kent State',
            'Long Beach State': 'Long Beach State',
            'Long Beach St.': 'Long Beach State',
            'Kennesaw State': 'Kennesaw State',
            'Kennesaw St.': 'Kennesaw State',
            'Morehead State': 'Morehead State',
            'Morehead St.': 'Morehead State',
            'Texas A&M-CC': 'Texas A&M-Corpus Christi',
            'Texas A&M-Corpus Christi': 'Texas A&M-Corpus Christi',
            'Texas Southern': 'Texas Southern',
            'Texas So': 'Texas Southern',
            'Cal State Fullerton': 'Cal State Fullerton',
            'CS Fullerton': 'Cal State Fullerton',
            'Grambling': 'Grambling State',
            'Grambling State': 'Grambling State',
            'McNeese State': 'McNeese State',
            'McNeese St.': 'McNeese State',
            'McNeese': 'McNeese State',
            'UAB': 'UAB',
            'Alabama-Birmingham': 'UAB',
            'Louisiana': 'Louisiana',
            'Louisiana-Lafayette': 'Louisiana',
            'Utah State': 'Utah State',
            'Utah St.': 'Utah State',
            'Charleston': 'College of Charleston',
            'College of Charleston': 'College of Charleston',
            'UC Irvine': 'UC Irvine',
            'UC-Irvine': 'UC Irvine',
            'Western Kentucky': 'Western Kentucky',
            'W Kentucky': 'Western Kentucky',
            'Boise State': 'Boise State',
            'Boise St.': 'Boise State',
            'Colorado State': 'Colorado State',
            'Colorado St.': 'Colorado State',
            'Eastern Washington': 'Eastern Washington',
            'E Washington': 'Eastern Washington',
        }

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

    def load_tournament_teams(self):
        """Load tournament teams from the tournament results data"""
        logger.info(f"Loading tournament teams from {self.tournament_data_path}")

        try:
            # Load tournament data
            tourney_df = pd.read_csv(self.tournament_data_path)

            # Extract unique teams for each year
            for year in range(self.start_year, self.end_year + 1):
                year_data = tourney_df[tourney_df['year'] == year]
                if len(year_data) == 0:
                    logger.warning(f"No tournament data found for {year}")
                    continue

                # Get unique teams from team1 and team2 columns
                teams = set(year_data['team1'].unique()).union(set(year_data['team2'].unique()))
                self.tournament_teams[year] = list(teams)

                logger.info(f"Found {len(teams)} teams for {year} tournament")

        except Exception as e:
            logger.error(f"Error loading tournament teams: {e}")
            return False

        return True

    def login_to_kenpom(self):
        """Log in to KenPom.com using provided credentials"""
        logger.info("Logging in to KenPom.com")

        try:
            self.browser = login(self.email, self.password)
            logger.info("Successfully logged in to KenPom.com")
            return True
        except Exception as e:
            logger.error(f"Error logging in to KenPom.com: {e}")
            return False

    def scrape_kenpom_data(self):
        """Scrape KenPom data for each year"""
        if self.browser is None:
            logger.error("Not logged in to KenPom.com. Please call login_to_kenpom() first.")
            return False

        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"Scraping KenPom data for {year}")
            year_str = str(year)

            try:
                # Get Pomeroy Ratings (overall ratings/rankings table)
                pomeroy_ratings = misc.get_pomeroy_ratings(self.browser, year_str)

                # Get efficiency stats
                efficiency_stats = summary.get_efficiency(self.browser, year_str)

                # Get Four Factors data
                fourfactors_data = summary.get_fourfactors(self.browser, year_str)

                # Get height/experience data
                height_data = summary.get_height(self.browser, year_str)

                # Get point distribution data
                pointdist_data = summary.get_pointdist(self.browser, year_str)

                # Store raw dataframes (before merging)
                year_data = {
                    'pomeroy_ratings': pomeroy_ratings,
                    'efficiency_stats': efficiency_stats,
                    'fourfactors_data': fourfactors_data,
                    'height_data': height_data,
                    'pointdist_data': pointdist_data
                }

                # Save individual raw data files
                for data_type, df in year_data.items():
                    output_file = os.path.join(self.output_dir, f"{data_type}_{year}.csv")
                    df.to_csv(output_file, index=False)
                    logger.info(f"Saved {data_type} data for {year} to {output_file}")

                # Store in our dictionary
                self.kenpom_data[year] = year_data

                # Merge data for this year (for easier team matching)
                merged_df = self._merge_year_data(year_data)

                # Save merged data
                output_file = os.path.join(self.output_dir, f"kenpom_merged_{year}.csv")
                merged_df.to_csv(output_file, index=False)
                logger.info(f"Saved merged KenPom data for {year} to {output_file}")

                # Be nice to the server
                time.sleep(3)

            except Exception as e:
                logger.error(f"Error scraping KenPom data for {year}: {e}")
                continue

        return True

    def _merge_year_data(self, year_data):
        """Merge all dataframes for a year into a single dataframe"""
        # Start with the main ratings
        merged_df = year_data['pomeroy_ratings'].copy()

        # Ensure a consistent team name column exists:
        if 'TeamName' not in merged_df.columns:
            if 'Team' in merged_df.columns:
                merged_df.rename(columns={'Team': 'TeamName'}, inplace=True)
            else:
                raise KeyError("Neither 'TeamName' nor 'Team' column found in pomeroy_ratings")

        # Define the team name column for each dataframe
        team_col_mappings = {
            'pomeroy_ratings': 'TeamName',
            'efficiency_stats': 'Team',
            'fourfactors_data': 'Team',
            'height_data': 'Team',
            'pointdist_data': 'Team'
        }

        # Merge with each additional dataframe
        for data_type, df in year_data.items():
            if data_type == 'pomeroy_ratings':
                continue  # Skip the base dataframe

            team_col = team_col_mappings.get(data_type, 'Team')

            # Clean team names to help with merging
            if team_col in df.columns:
                df[team_col] = df[team_col].str.replace(r'^\d+\s+', '', regex=True)

                merged_df = pd.merge(
                    merged_df,
                    df,
                    left_on='TeamName',
                    right_on=team_col,
                    how='outer',
                    suffixes=('', f'_{data_type}')
                )

                # Remove duplicate team columns if they exist
                if team_col != 'TeamName' and team_col in merged_df.columns:
                    merged_df.drop(columns=[team_col], inplace=True)

        return merged_df

    def extract_tournament_team_stats(self):
        """Extract KenPom stats only for tournament teams"""
        if not self.tournament_teams or not self.kenpom_data:
            logger.error("Tournament teams or KenPom data not loaded. Please load data first.")
            return False

        # Create a dictionary to store tournament team stats by year
        tournament_stats = {}

        for year in range(self.start_year, self.end_year + 1):
            if year not in self.tournament_teams:
                logger.warning(f"Tournament teams for {year} not found. Skipping.")
                continue

            if year not in self.kenpom_data:
                logger.warning(f"KenPom data for {year} not found. Skipping.")
                continue

            # Load merged data for this year
            merged_file = os.path.join(self.output_dir, f"kenpom_merged_{year}.csv")
            if not os.path.exists(merged_file):
                logger.warning(f"Merged KenPom data file not found: {merged_file}. Skipping.")
                continue

            merged_df = pd.read_csv(merged_file)

            # Get tournament teams for this year
            teams = self.tournament_teams[year]

            # Filter for tournament teams
            tournament_team_stats = []
            unmatched_teams = []

            for team in teams:
                # Try direct name match first
                team_data = merged_df[merged_df['TeamName'] == team]

                # If no match, try standardized name
                if len(team_data) == 0 and team in self.team_name_mapping:
                    kenpom_name = self.team_name_mapping[team]
                    team_data = merged_df[merged_df['TeamName'] == kenpom_name]

                # If still no match, try substring match
                if len(team_data) == 0:
                    # Try partial matching (contains)
                    team_data = merged_df[merged_df['TeamName'].str.contains(team, case=False, na=False)]

                    # Special case for teams like "NC State" vs "North Carolina State"
                    if len(team_data) == 0 and "State" in team:
                        state_name = team.split(" ")[0]
                        team_data = merged_df[merged_df['TeamName'].str.contains(f"{state_name}.*State", case=False, regex=True, na=False)]

                if len(team_data) == 0:
                    unmatched_teams.append(team)
                    continue

                if len(team_data) > 1:
                    logger.warning(f"Multiple matches found for {team} in {year}. Using first match.")

                # Get the first match
                team_row = team_data.iloc[0].copy()

                # Add tournament team name for joining
                team_row['TournamentTeam'] = team

                # Add to list
                tournament_team_stats.append(team_row)

            if unmatched_teams:
                logger.warning(f"Unmatched teams in {year}: {unmatched_teams}")

            # Create DataFrame for this year's tournament teams
            if tournament_team_stats:
                tournament_stats[year] = pd.DataFrame(tournament_team_stats)

                # Save tournament team stats for this year
                output_file = os.path.join(self.output_dir, f"kenpom_tournament_teams_{year}.csv")
                tournament_stats[year].to_csv(output_file, index=False)
                logger.info(f"Saved KenPom stats for {len(tournament_team_stats)} tournament teams in {year} to {output_file}")

        # Create a combined file with all years
        if tournament_stats:
            all_years_df = pd.concat([tournament_stats[year] for year in tournament_stats.keys()])
            all_years_file = os.path.join(self.output_dir, "kenpom_all_tournament_teams.csv")
            all_years_df.to_csv(all_years_file, index=False)
            logger.info(f"Saved combined KenPom stats for all tournament teams to {all_years_file}")

        return True

    def create_matchup_features(self):
        """Create matchup features for tournament games"""
        logger.info("Creating matchup features for tournament games")

        try:
            # Load tournament data
            tourney_df = pd.read_csv(self.tournament_data_path)

            # Load all KenPom tournament team stats
            kenpom_file = os.path.join(self.output_dir, "kenpom_all_tournament_teams.csv")
            if not os.path.exists(kenpom_file):
                logger.error(f"KenPom tournament team stats file not found: {kenpom_file}")
                return None

            kenpom_df = pd.read_csv(kenpom_file)

            # Add year column if it doesn't exist
            if 'Year' not in kenpom_df.columns and 'year' in kenpom_df.columns:
                kenpom_df['Year'] = kenpom_df['year']
            elif 'Year' not in kenpom_df.columns and 'Season' in kenpom_df.columns:
                kenpom_df['Year'] = kenpom_df['Season']
            elif 'Year' not in kenpom_df.columns:
                logger.error("Could not find year column in KenPom data")
                return None

            # Prepare a list to store all matchup features
            all_matchups = []

            # Process each year
            for year in range(self.start_year, self.end_year + 1):
                # Get tournament games for this year
                year_tourney = tourney_df[tourney_df['year'] == year]

                # Get KenPom data for this year
                year_kenpom = kenpom_df[kenpom_df['Year'] == year]

                if len(year_tourney) == 0 or len(year_kenpom) == 0:
                    logger.warning(f"Missing data for {year}. Skipping.")
                    continue

                # Process each game
                for _, game in year_tourney.iterrows():
                    team1 = game['team1']
                    team2 = game['team2']

                    # Get KenPom stats for both teams
                    team1_stats = year_kenpom[year_kenpom['TournamentTeam'] == team1]
                    team2_stats = year_kenpom[year_kenpom['TournamentTeam'] == team2]

                    if len(team1_stats) == 0 or len(team2_stats) == 0:
                        logger.warning(f"Missing KenPom stats for game: {team1} vs {team2} in {year}. Skipping.")
                        continue

                    # Create matchup features
                    matchup = {
                        'Year': year,
                        'Round': game['round'],
                        'Team1': team1,
                        'Team2': team2,
                        'Team1_Seed': game['team1_seed'],
                        'Team2_Seed': game['team2_seed'],
                        'Winner': game['winner']
                    }

                    # Get first row of each team's stats
                    t1_stats = team1_stats.iloc[0]
                    t2_stats = team2_stats.iloc[0]

                    # Key KenPom metrics to include
                    key_metrics = [
                        'Rk', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SOS AdjEM', 'OppO', 'OppD', 'NCSOS AdjEM',
                        'EFG%', 'EFG%D', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD',
                        'Avg. Ht.', 'Effective Ht.', 'Continuity', 'Experience', '2P%', '3P%', 'FT%'
                    ]

                    # Add individual team metrics
                    for metric in key_metrics:
                        # Check if metric exists in the data (column names may vary)
                        if metric in t1_stats and metric in t2_stats:
                            # Convert to float if possible
                            try:
                                t1_value = float(t1_stats[metric])
                                t2_value = float(t2_stats[metric])

                                matchup[f'Team1_{metric.replace("%", "Pct").replace(".", "").replace(" ", "")}'] = t1_value
                                matchup[f'Team2_{metric.replace("%", "Pct").replace(".", "").replace(" ", "")}'] = t2_value

                                # Add differential metrics
                                if 'Rk' in metric or 'Rank' in metric:
                                    # For ranks, lower is better, so subtract in reverse order
                                    matchup[f'{metric.replace("%", "Pct").replace(".", "").replace(" ", "")}_Diff'] = t2_value - t1_value
                                else:
                                    matchup[f'{metric.replace("%", "Pct").replace(".", "").replace(" ", "")}_Diff'] = t1_value - t2_value
                            except (ValueError, TypeError):
                                # Skip if can't convert to float
                                pass

                    # Add seed difference
                    if pd.notna(game['team1_seed']) and pd.notna(game['team2_seed']):
                        matchup['Seed_Diff'] = game['team1_seed'] - game['team2_seed']

                    # Add target variable: 1 if Team1 won, 0 if Team2 won
                    matchup['Team1_Won'] = 1 if game['winner'] == team1 else 0

                    all_matchups.append(matchup)

            # Create DataFrame from all matchups
            matchups_df = pd.DataFrame(all_matchups)

            # Fill NaN values
            matchups_df = matchups_df.fillna(0)

            # Save matchup features
            output_file = os.path.join(self.output_dir, "tournament_matchup_features.csv")
            matchups_df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(matchups_df)} tournament matchup features to {output_file}")

            return matchups_df

        except Exception as e:
            logger.error(f"Error creating matchup features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def run_full_pipeline(self):
        """Run the full data collection pipeline"""
        logger.info("Starting full KenPom data collection pipeline")

        # Step 1: Load tournament teams
        if not self.load_tournament_teams():
            logger.error("Failed to load tournament teams. Exiting.")
            return False

        # Step 2: Login to KenPom
        if not self.login_to_kenpom():
            logger.error("Failed to login to KenPom. Exiting.")
            return False

        # Step 3: Scrape KenPom data
        if not self.scrape_kenpom_data():
            logger.error("Failed to scrape KenPom data. Exiting.")
            return False

        # Step 4: Extract tournament team stats
        if not self.extract_tournament_team_stats():
            logger.error("Failed to extract tournament team stats. Exiting.")
            return False

        # Step 5: Create matchup features
        matchups_df = self.create_matchup_features()
        if matchups_df is None:
            logger.error("Failed to create matchup features. Exiting.")
            return False

        logger.info("Successfully completed KenPom data collection pipeline")
        return True


def main():
    # Interactive input instead of command-line arguments
    from getpass import getpass

    email = input('Enter your KenPom email: ')
    password = getpass('Enter your KenPom password: ')
    start_year = int(input('Enter start year (default 2011): ') or 2015)
    end_year = int(input('Enter end year (default 2024): ') or 2025)
    tournament_data = input('Enter tournament data path (default "data\tournament_data\ncaa_tournament_results_simplified.csv"): ') or r"data\tournament_data\ncaa_tournament_results_simplified.csv"
    output_dir = input('Enter output directory (default "data\kenpom_data"): ') or r"data\kenpom_data"

    collector = KenPomDataCollector(
        email=email,
        password=password,
        start_year=start_year,
        end_year=end_year,
        tournament_data_path=tournament_data,
        output_dir=output_dir
    )

    collector.run_full_pipeline()

# Make sure this line is at the end of your file
if __name__ == "__main__":
    main()