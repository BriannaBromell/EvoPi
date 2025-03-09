# game_state.py
import pickle
import json
import os
from pathlib import Path

def save_game_state(game_data, filename="game_state.pkl"):
    """Save game state with JSON metadata"""
    try:
        # Create saved_data directory if needed
        saved_dir = Path("saved_data")
        saved_dir.mkdir(exist_ok=True)
        
        # Save JSON metadata with data keys
        with open(saved_dir / "game_metadata.json", "w") as f:
            json.dump({"data_keys": list(game_data.keys())}, f)
        
        # Save pickle data
        with open(saved_dir / filename, "wb") as f:
            pickle.dump(game_data, f)
            
        print(f"Game state saved to {saved_dir/filename}")
        return True
    except Exception as e:
        print(f"Error saving game state: {e}")
        return False

def load_game_state(filename="game_state.pkl"):
    """Load game state using JSON metadata"""
    try:
        saved_dir = Path("saved_data")
        
        # Load JSON metadata
        with open(saved_dir / "game_metadata.json", "r") as f:
            metadata = json.load(f)
            
        # Load pickle data
        with open(saved_dir / filename, "rb") as f:
            loaded_data = pickle.load(f)
        
        # Create complete game state with all expected keys
        game_state = {}
        for key in metadata["data_keys"]:
            game_state[key] = loaded_data.get(key)
            
        return game_state
    except FileNotFoundError:
        print("No saved game state found")
        return None
    except Exception as e:
        print(f"Error loading game state: {e}")
        return None