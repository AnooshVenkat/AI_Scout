import time
import json
import os
import shutil
import chromadb
import requests
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from sentence_transformers import SentenceTransformer
from datetime import datetime

PROGRESS_FILE = 'progress.json'
CSV_DATABASE = 'all_games.csv'
DB_PATH = "./chroma_db"

CSV_HEADERS = [
    'player_id', 'player_name', 'season', 'game_id', 'opponent', 'pts', 'reb', 'ast', 
    'plus_minus', 'blk', 'stl', 'tov', 'pf', 'fgm', 'fga', 'fg3m', 'fg3a'
]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try: return json.load(f)['next_player_index']
            except (json.JSONDecodeError, KeyError): return 0
    return 0

def save_progress(index):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'next_player_index': index}, f)

def fetch_player_data(player_id, player_name):
    all_docs, all_metadatas, structured_games = [], [], []
    try:
        player_info_df = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=60).get_data_frames()[0]
        from_year, to_year_from_api = int(player_info_df['FROM_YEAR'].iloc[0]), int(player_info_df['TO_YEAR'].iloc[0])
        
        current_year, current_month = datetime.now().year, datetime.now().month
        latest_season_start_year = current_year if current_month >= 10 else current_year - 1
        effective_to_year = min(to_year_from_api, latest_season_start_year)

        active_seasons = []
        if from_year <= effective_to_year:
            active_seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(from_year, effective_to_year + 1)]

        if not active_seasons: return [], [], []
        
        print(f"    - Fetching {len(active_seasons)} valid season(s)...")

        for season in active_seasons:
            for attempt in range(3):
                try:
                    time.sleep(1)
                    game_log_df = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=60).get_data_frames()[0]
                    if not game_log_df.empty:
                        for _, row in game_log_df.iterrows():
                            all_docs.append(f"In {season}, vs {row['MATCHUP'].split(' ')[-1]}, {player_name} had {row['PTS']}p, {row['REB']}r, {row['AST']}a, {row['STL']}s, {row['BLK']}b. +/- was {row['PLUS_MINUS']}.")
                            all_metadatas.append({"player_name": player_name, "season": season, "opponent": row['MATCHUP'].split(' ')[-1]})
                            structured_games.append({h: row.get(h.upper()) for h in CSV_HEADERS})
                            structured_games[-1].update({'player_id': player_id, 'player_name': player_name, 'season': season, 'game_id': row['Game_ID'], 'opponent': row['MATCHUP'].split(' ')[-1]})
                    break 
                except requests.exceptions.RequestException as e:
                    wait_time = 10 * (attempt + 1)
                    print(f"    - Network error for {season}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            else:
                print(f"    - Failed to fetch data for season {season} after multiple retries.")
        
        return all_docs, all_metadatas, structured_games
    except Exception as e:
        print(f"    - A critical error occurred for {player_name}: {e}")
        return [], [], []

def main():
    """Main function to build all databases from scratch, with corrected API calls."""
    print("Starting Advanced Data Ingestion...")
    
    try:
        all_players = players.get_active_players()
    except Exception as e:
        print(f"CRITICAL FAILURE: Could not get the initial list of players from the NBA API: {e}")
        return

    start_index = load_progress()
    if start_index == 0 and not os.path.exists(PROGRESS_FILE):
        print("First run detected. Setting up fresh databases...")
        if os.path.exists(CSV_DATABASE): os.remove(CSV_DATABASE)
        if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
        pd.DataFrame(columns=CSV_HEADERS).to_csv(CSV_DATABASE, index=False)
        save_progress(0)
        print("Fresh 'progress.json' and 'all_games.csv' created.")
    
    start_index = load_progress() 
    
    print(f"Resuming from player {start_index + 1} of {len(all_players)}.")
    
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="nba_scout")
    doc_counter = collection.count()

    for i in range(start_index, len(all_players)):
        player = all_players[i]
        print(f"\n({i+1}/{len(all_players)}) Processing: {player['full_name']}...")
        
        documents, metadatas, games = fetch_player_data(player['id'], player['full_name'])
        
        if games:
            if documents:
                embeddings = model.encode(documents, show_progress_bar=False)
                ids = [f"doc_{doc_counter + j}" for j in range(len(documents))]
                collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
                doc_counter += len(documents)
            
            df = pd.DataFrame(games)
            df.to_csv(CSV_DATABASE, mode='a', header=False, index=False)
            print(f"    - Added {len(documents)} docs to Vector DB and {len(games)} records to CSV.")
        else:
            print(f"    - No new game data found. Skipping.")
        
        save_progress(i + 1)
            
    print("\n All players have been processed!")

if __name__ == "__main__":
    main()