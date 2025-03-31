import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"ncaa_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NCAABracketScraper:
    """
    Scraper for NCAA tournament data from Sports-Reference.com
    """

    def __init__(self, start_year, end_year, output_dir=r"data\tournament_data"):
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir
        self.base_url = "https://www.sports-reference.com/cbb/postseason/men/{}-ncaa.html"
        self.all_games = []
        self.all_teams = set()

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        # Round names and corresponding numbers
        self.round_mapping = {
            'First Four': 0,
            'First Round': 1,
            'Round of 64': 1,  # Alternative name
            'Second Round': 2,
            'Round of 32': 2,  # Alternative name
            'Sweet 16': 3,
            'Regional Semifinals': 3,  # Alternative name
            'Elite 8': 4,
            'Regional Finals': 4,  # Alternative name
            'Final Four': 5,
            'National Semifinals': 5,  # Alternative name
            'Championship': 6,
            'National Championship': 6  # Alternative name
        }

        # Team name standardization mapping
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
            'Florida Gulf Coast.': 'Florida Gulf Coast.',
            'FGCU': 'Florida Gulf Coast.',
            'Fla Gulf Coast.': 'Florida Gulf Coast.',
            'FL Gulf Coast.': 'Florida Gulf Coast.',
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

        # Setup headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }

    def scrape_tournament(self, year):
        """Scrape NCAA tournament data for a specific year"""
        url = self.base_url.format(year)
        logger.info(f"Scraping {year} tournament data from {url}")

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the tabs for regions
            region_tabs = soup.select('.switcher.filter[data-controls="#brackets"] div')
            if not region_tabs:
                logger.warning(f"Could not find region tabs for {year}. Trying alternate method...")
                # Try alternate method for older years where structure may be different
                region_divs = soup.select('#brackets > div')
                if region_divs:
                    regions = []
                    for div in region_divs:
                        region_id = div.get('id')
                        if region_id:
                            regions.append(region_id)

                    if regions:
                        logger.info(f"Found regions using alternate method: {regions}")
                        for region in regions:
                            region_div = soup.find('div', id=region)
                            if region_div:
                                self._process_region(region_div, region, year)
                        return True
                logger.warning(f"Could not find regions for {year}")
                return False

            # Get region names
            regions = []
            for tab in region_tabs:
                region_name = tab.get_text(strip=True)
                if region_name:
                    regions.append(region_name)

            logger.info(f"Found regions for {year}: {regions}")

            # Find bracket contents
            brackets_div = soup.find('div', id='brackets')
            if not brackets_div:
                logger.warning(f"Could not find brackets div for {year}")
                return False

            # Process each region
            for i, region in enumerate(regions):
                region_id = region.lower()
                region_div = brackets_div.find('div', id=region_id)

                if not region_div:
                    # Try alternate ID format
                    for div in brackets_div.find_all('div', class_=lambda c: c and 'current' in c):
                        if i == 0 and not div.get('id'):
                            region_div = div
                            break

                if not region_div:
                    logger.warning(f"Could not find bracket for region {region} in {year}")
                    continue

                # Process this region
                self._process_region(region_div, region, year)

            logger.info(f"Successfully scraped {year} tournament data")
            return True

        except Exception as e:
            logger.error(f"Error scraping {year} tournament: {e}")
            return False

    def _process_region(self, region_div, region_name, year):
        """Process games in a tournament region"""
        logger.info(f"Processing {region_name} region for {year}")

        # Find all rounds in this region
        rounds = region_div.find_all('div', class_='round')

        for round_idx, round_div in enumerate(rounds):
            # Determine round number (0 = First Four, 1 = First Round, etc.)
            if region_name.lower() == 'national':
                # National rounds start at Final Four (round 5)
                round_num = 5 + round_idx if round_idx < 2 else None
            else:
                # Regular regions
                round_num = round_idx + 1  # 1-indexed (First Round = 1)
                if round_num > 4:  # Beyond Elite 8 doesn't belong to a region
                    continue

            # Process each game in this round
            games = round_div.find_all('div', recursive=False)
            for game in games:
                # Skip if this is not actually a game div
                if not game.find_all('div', recursive=False):
                    continue

                # Find team divs
                team_divs = game.find_all('div', recursive=False)
                if len(team_divs) < 2:
                    continue

                # Get teams
                team1_div = team_divs[0]
                team2_div = team_divs[1]

                # Extract team info
                team1_data = self._extract_team_info(team1_div)
                team2_data = self._extract_team_info(team2_div)

                if not team1_data or not team2_data:
                    continue

                # Standardize team names
                team1_data['name'] = self._standardize_team_name(team1_data['name'])
                team2_data['name'] = self._standardize_team_name(team2_data['name'])

                # Add to all teams we've seen
                self.all_teams.add(team1_data['name'])
                self.all_teams.add(team2_data['name'])

                # Determine winner
                winner = None
                if team1_div.get('class') and 'winner' in team1_div.get('class'):
                    winner = team1_data['name']
                elif team2_div.get('class') and 'winner' in team2_div.get('class'):
                    winner = team2_data['name']

                # Skip future games (no winner)
                if not winner:
                    continue

                # Add game to results
                self.all_games.append({
                    'year': year,
                    'region': region_name,
                    'round': round_num,
                    'team1': team1_data['name'],
                    'team2': team2_data['name'],
                    'team1_seed': team1_data['seed'],
                    'team2_seed': team2_data['seed'],
                    'team1_score': team1_data['score'],
                    'team2_score': team2_data['score'],
                    'winner': winner
                })

    def _extract_team_info(self, team_div):
        """Extract team name, seed, and score from a team div"""
        result = {
            'name': None,
            'seed': None,
            'score': None
        }

        # Extract seed
        seed_span = team_div.find('span')
        if seed_span:
            try:
                result['seed'] = int(seed_span.get_text(strip=True))
            except (ValueError, TypeError):
                pass

        # Extract team name
        team_links = team_div.find_all('a')
        if len(team_links) > 0:
            result['name'] = team_links[0].get_text(strip=True)

            if len(team_links) > 1:
                try:
                    result['score'] = int(team_links[1].get_text(strip=True))
                except (ValueError, TypeError):
                    pass

        return result if result['name'] else None

    def _standardize_team_name(self, team_name):
        """Standardize team name for consistency"""
        if not team_name:
            return team_name

        # Check direct mapping first
        if team_name in self.team_name_mapping:
            return self.team_name_mapping[team_name]

        # No direct match found, return original name
        return team_name

    def scrape_all_tournaments(self):
        """Scrape all tournaments in the specified year range"""
        for year in range(self.start_year, self.end_year + 1):
            success = self.scrape_tournament(year)

            if success:
                time.sleep(3)  # 3-second delay between years

    def validate_data(self):
        """Validate the scraped data"""
        if not self.all_games:
            logger.warning("No data to validate!")
            return

        df = pd.DataFrame(self.all_games)

        # Check for missing values in critical columns
        missing_values = df[['year', 'round', 'team1', 'team2', 'winner']].isnull().sum()
        logger.info("Missing values in critical columns:")
        logger.info(missing_values)

        # Check for expected number of games per year
        games_per_year = df.groupby('year').size()
        logger.info("\nGames per year:")
        for year, count in games_per_year.items():
            # Before 2011, tournaments had 65 teams (1 play-in game)
            # From 2011 onwards, tournaments have 68 teams (4 play-in games)
            expected_min = 63  # Main bracket without play-in games
            expected_max = 67 if year >= 2011 else 64  # Including play-in games

            if count < expected_min or count > expected_max:
                logger.warning(f"Year {year} has {count} games (expected between {expected_min} and {expected_max})")
            else:
                logger.info(f"Year {year} has {count} games (expected between {expected_min} and {expected_max})")

        # Check that winners are either team1 or team2
        invalid_winners = df[~df.apply(lambda row: row['winner'] in [row['team1'], row['team2']], axis=1)]
        if not invalid_winners.empty:
            logger.warning(f"\nWarning: Found {len(invalid_winners)} games with invalid winners")
            logger.warning(invalid_winners[['year', 'round', 'team1', 'team2', 'winner']].head())

        # Print some statistics about the data
        logger.info(f"\nTotal games scraped: {len(df)}")
        logger.info(f"Total unique teams: {len(self.all_teams)}")
        logger.info(f"Years covered: {df['year'].min()} to {df['year'].max()}")
        logger.info(f"Rounds covered: {df['round'].min()} to {df['round'].max()}")

    def save_to_csv(self):
        """Save the scraped data to CSV files"""
        if not self.all_games:
            logger.warning("No data to save!")
            return

        # Create DataFrame
        df = pd.DataFrame(self.all_games)

        # Sort by year, round, region
        df = df.sort_values(['year', 'round', 'region'])

        # Create a simplified version with just required columns for modeling
        simplified_df = df[['year', 'round', 'team1', 'team2', 'team1_seed', 'team2_seed', 'winner']].copy()
        simplified_file = os.path.join(self.output_dir, "ncaa_tournament_results_simplified.csv")
        simplified_df.to_csv(simplified_file, index=False)
        logger.info(f"Saved simplified data to {simplified_file}")

        # Save a list of all teams seen
        teams_file = os.path.join(self.output_dir, "all_tournament_teams.txt")
        with open(teams_file, 'w') as f:
            for team in sorted(self.all_teams):
                f.write(f"{team}\n")
        logger.info(f"Saved list of {len(self.all_teams)} teams to {teams_file}")

        logger.info("Data saving complete!")

# Main execution
if __name__ == "__main__":
    scraper1 = NCAABracketScraper(start_year=2020, end_year=2024)
    scraper1.scrape_all_tournaments()
    scraper1.validate_data()
    scraper1.save_to_csv()