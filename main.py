import os
import pandas as pd
import openai # Use the OpenAI library
import chromadb
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CSV_DATABASE = 'all_games.csv'

STAT_MAP = {
    'points': 'pts', 'pts': 'pts', 'rebounds': 'reb', 'reb': 'reb', 'assists': 'ast', 'ast': 'ast',
    'blocks': 'blk', 'blk': 'blk', 'steals': 'stl', 'stl': 'stl', 'turnovers': 'tov', 'tov': 'tov',
    'fouls': 'pf', 'pf': 'pf', 'plus-minus': 'plus_minus', 'plus_minus': 'plus_minus'
}
TEAM_NAME_MAP = {
    '76ers': 'PHI', 'sixers': 'PHI', 'philadelphia': 'PHI', 'bucks': 'MIL', 'celtics': 'BOS', 'nets': 'BKN',
    'knicks': 'NYK', 'raptors': 'TOR', 'bulls': 'CHI', 'cavaliers': 'CLE', 'pacers': 'IND', 'pistons': 'DET',
    'heat': 'MIA', 'hawks': 'ATL', 'hornets': 'CHA', 'magic': 'ORL', 'wizards': 'WAS', 'nuggets': 'DEN',
    'timberwolves': 'MIN', 'thunder': 'OKC', 'blazers': 'POR', 'jazz': 'UTA', 'warriors': 'GSW',
    'clippers': 'LAC', 'lakers': 'LAL', 'suns': 'PHX', 'kings': 'SAC', 'grizzlies': 'MEM', 'mavericks': 'DAL',
    'rockets': 'HOU', 'pelicans': 'NOP', 'spurs': 'SAS'
}
CALCULABLE_STATS = ['pts', 'reb', 'ast', 'blk', 'stl', 'tov', 'plus_minus']

def calculate_player_averages(player_name: str, seasons: str = "", opponent: str = "") -> str:
    try:
        df = pd.read_csv(CSV_DATABASE)
        filtered_df = df[df['player_name'].str.lower() == player_name.lower()]
        if seasons:
            filtered_df = filtered_df[filtered_df['season'].isin([s.strip() for s in seasons.split(',')])]
        if opponent:
            opponent_abbr = TEAM_NAME_MAP.get(opponent.lower(), opponent.upper())
            filtered_df = filtered_df[filtered_df['opponent'] == opponent_abbr]
        if filtered_df.empty: return "No game data found for the specified criteria."
        
        avg_stats = {stat: filtered_df[stat].mean() for stat in CALCULABLE_STATS}
        
        total_fga = filtered_df['fga'].sum()
        avg_stats['fg_percentage'] = 0 if total_fga == 0 else (filtered_df['fgm'].sum() / total_fga) * 100
        total_fg3a = filtered_df['fg3a'].sum()
        avg_stats['fg3_percentage'] = 0 if total_fg3a == 0 else (filtered_df['fg3m'].sum() / total_fg3a) * 100

        return json.dumps({
            "games_found": len(filtered_df),
            "averages": {k: round(v, 1) for k, v in avg_stats.items()}
        })
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_player_season_info(player_name: str) -> str:
    try:
        df = pd.read_csv(CSV_DATABASE)
        player_df = df[df['player_name'].str.lower() == player_name.lower()]
        if player_df.empty: return "No data found for this player."
        seasons_played = player_df['season'].unique().tolist()
        return json.dumps({"player_name": player_name, "season_count": len(seasons_played), "seasons_played": seasons_played})
    except Exception as e: return f"An error occurred: {str(e)}"

def compare_players_averages(player_a_name: str, player_b_name: str, seasons: str = "") -> str:
    try:
        df = pd.read_csv(CSV_DATABASE)
        results = {}
        for player_name in [player_a_name, player_b_name]:
            filtered_df = df[df['player_name'].str.lower() == player_name.lower()]
            if seasons:
                filtered_df = filtered_df[filtered_df['season'].isin([s.strip() for s in seasons.split(',')])]
            if filtered_df.empty:
                results[player_name] = "No data found."
                continue
            
            avg_stats = {stat: filtered_df[stat].mean() for stat in CALCULABLE_STATS}
            results[player_name] = {k: round(v, 1) for k, v in avg_stats.items()}
        return json.dumps(results)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_player_career_high(player_name: str, stat: str) -> str:
    try:
        stat_column = STAT_MAP.get(stat.lower())
        if not stat_column:
            return f"Invalid stat '{stat}'. Please use a supported statistic."
        df = pd.read_csv(CSV_DATABASE)
        player_df = df[df['player_name'].str.lower() == player_name.lower()]
        if player_df.empty: return "No data found for this player."
        career_high_game = player_df.loc[player_df[stat_column].idxmax()]
        return json.dumps({
            "player_name": player_name, "career_high_stat": stat,
            "stat_value": int(career_high_game[stat_column]), "season": career_high_game['season'],
            "opponent": career_high_game['opponent'], "game_id": career_high_game['game_id']
        })
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_player_stat_progression(player_name: str, stats: str) -> str:
    try:
        stat_list_raw = [s.strip().lower() for s in stats.split(',')]
        stat_columns_map = {s: STAT_MAP.get(s) for s in stat_list_raw}
        invalid_stats = [s for s, c in stat_columns_map.items() if c is None]
        if invalid_stats:
            return f"Invalid stat(s) provided: {', '.join(invalid_stats)}."
        df = pd.read_csv('all_games.csv')
        player_df = df[df['player_name'].str.lower() == player_name.lower()]
        if player_df.empty: return f"No data found for player '{player_name}'."
        results = {"player_name": player_name, "progression_data": {}}
        for user_stat, col_name in stat_columns_map.items():
            if col_name in player_df.columns:
                progression = player_df.groupby('season')[col_name].mean().round(1)
                career_average = player_df[col_name].mean()
                peak_season = progression.idxmax()
                peak_value = progression.max()
                results["progression_data"][user_stat] = {
                    "career_average": round(career_average, 1),
                    "peak_season": peak_season,
                    "peak_value": peak_value,
                    "season_by_season": progression.to_dict()
                }
        return json.dumps(results)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def find_top_performer_against_team(opponent_team: str, stat: str = "points", season: str = "") -> str:
    try:
        opponent_abbr = TEAM_NAME_MAP.get(opponent_team.lower())
        if not opponent_abbr and opponent_team.upper() in TEAM_NAME_MAP.values():
            opponent_abbr = opponent_team.upper()
        if not opponent_abbr: return f"Could not find the team '{opponent_team}'."
        stat_col = STAT_MAP.get(stat.lower())
        if not stat_col: return f"Invalid stat '{stat}'."
        df = pd.read_csv(CSV_DATABASE)
        filtered_df = df[df['opponent'] == opponent_abbr]
        if season:
            filtered_df = filtered_df[filtered_df['season'] == season]
        if filtered_df.empty: return f"No game data found against {opponent_team} for the specified criteria."
        player_averages = filtered_df.groupby('player_name')[stat_col].mean()
        if player_averages.empty: return f"Could not compute averages against {opponent_team}."
        top_performer_name = player_averages.idxmax()
        top_average_value = player_averages.max()
        top_performer_games_df = filtered_df[filtered_df['player_name'] == top_performer_name]
        games_played = len(top_performer_games_df)
        single_game_high = int(top_performer_games_df[stat_col].max())
        return json.dumps({
            "top_performer": top_performer_name, "against_team": opponent_team, "in_season": season if season else "All-Time", 
            "stat": stat, "average_value": round(top_average_value, 1), 
            "games_played": games_played, "single_game_high": single_game_high
        })
    except Exception as e:
        return f"An error occurred during analysis: {str(e)}"
    
def get_player_total_stats(player_name: str, seasons: str = "", game_type: str = "") -> str:
    try:
        df = pd.read_csv('all_games.csv')
        filtered_df = df[df['player_name'].str.lower() == player_name.lower()]
        if game_type:
            filtered_df = filtered_df[filtered_df['game_type'].str.lower() == game_type.lower()]
        if seasons:
            season_list = [s.strip() for s in seasons.split(',')]
            filtered_df = filtered_df[filtered_df['season'].isin(season_list)]
        if filtered_df.empty:
            return "No game data found for the specified criteria."
        total_stats = {stat: int(filtered_df[stat].sum()) for stat in CALCULABLE_STATS}
        total_stats['fgm'] = int(filtered_df['fgm'].sum())
        total_stats['fga'] = int(filtered_df['fga'].sum())
        total_stats['fg3m'] = int(filtered_df['fg3m'].sum())
        total_stats['fg3a'] = int(filtered_df['fg3a'].sum())
        return json.dumps({
            "games_found": len(filtered_df),
            "totals": total_stats
        })
    except Exception as e:
        return f"An error occurred: {str(e)}"

available_tools = {
    "calculate_player_averages": calculate_player_averages,
    "get_player_season_info": get_player_season_info,
    "compare_players_averages": compare_players_averages,
    "get_player_career_high": get_player_career_high,
    "get_player_stat_progression": get_player_stat_progression,
    "find_top_performer_against_team": find_top_performer_against_team,
    "get_player_total_stats": get_player_total_stats,
}

system_prompt = """
You are an expert NBA scout and data analyst. Your primary role is to answer user questions by calling your available tools, then synthesizing the data into a professional scouting report.
CRITICAL RULE 1: The data returned from your tools is the absolute source of truth for all statistics.
CRITICAL RULE 2: Format your final response using Markdown for maximum readability. Use headings, **bold text** for key terms, and bullet points.
CRITICAL RULE 3: When a tool provides season-by-season data (like a stat progression), your output MUST follow this specific two-part structure:
1.  **The Table:** First, present a single, combined Markdown table of the raw season-by-season data.
2.  **Scouting Report:** Second, below the table, provide a narrative analysis titled 'Scouting Report'. For each statistic, you must perform a two-step analysis:
    a. Briefly describe the numerical trend based on the data in the table (e.g., 'His steals per game peaked in the 2018-19 season...').
    b. Use your broader basketball knowledge to explain what these numbers mean. Contextualize the stats. Is an average of 1.6 steals elite for a guard? What does it imply about their defensive style, IQ, or role on the team? This is where you provide real scouting insights beyond the numbers.
"""

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculate_player_averages",
            "description": "Calculates a player's detailed average stats including points, rebounds, assists, blocks, steals, turnovers, plus-minus, and shooting percentages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "The full name of the player."},
                    "seasons": {"type": "string", "description": "Optional. A comma-separated string of seasons (e.g., '2023-24')."},
                    "opponent": {"type": "string", "description": "Optional. The name or abbreviation of an opponent team."}
                },
                "required": ["player_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_season_info",
            "description": "Looks up a player and returns the total number of seasons played and a list of those seasons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "The full name of the player to look up."}
                },
                "required": ["player_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_players_averages",
            "description": "Compares detailed average stats between two players for their careers or specific seasons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_a_name": {"type": "string", "description": "The full name of the first player."},
                    "player_b_name": {"type": "string", "description": "The full name of the second player."},
                    "seasons": {"type": "string", "description": "Optional. A comma-separated string of seasons to compare."}
                },
                "required": ["player_a_name", "player_b_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_career_high",
            "description": "Finds a player's single-game career high for a specific statistic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "The full name of the player."},
                    "stat": {"type": "string", "description": "The statistic to find the career high for (e.g., 'points', 'rebounds', 'assists')."}
                },
                "required": ["player_name", "stat"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_stat_progression",
            "description": "Shows the season-by-season progression for one or more statistics for a player.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "The full name of the player."},
                    "stats": {"type": "string", "description": "A comma-separated string of statistics to track (e.g., 'points, assists')."}
                },
                "required": ["player_name", "stats"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_top_performer_against_team",
            "description": "Finds the player with the highest average for a specific stat against a given opponent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "opponent_team": {"type": "string", "description": "The name, city, or abbreviation of the opponent team."},
                    "stat": {"type": "string", "description": "Optional. The statistic to measure. Defaults to 'points'."},
                    "season": {"type": "string", "description": "Optional. The specific season to filter for."}
                },
                "required": ["opponent_team"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_total_stats",
            "description": "Calculates a player's total accumulated stats for given seasons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "The full name of the player."},
                    "seasons": {"type": "string", "description": "Optional. A comma-separated string of seasons."},
                    "game_type": {"type": "string", "description": "Optional. Filter for 'Regular Season' or 'Playoffs'."}
                },
                "required": ["player_name"]
            }
        }
    }
]

app = FastAPI()
class Query(BaseModel): query: str

@app.post("/scout")
async def scout_player(query: Query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query.query}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
            
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            return {"response": final_response.choices[0].message.content}
        else:
            return {"response": response_message.content}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred with the AI model.")

@app.get("/")
def read_root():
    return FileResponse('index.html')
