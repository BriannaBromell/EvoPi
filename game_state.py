import pickle
from game_gc import clean_game_state

def save_game_state(game_data, filename="game_state.pkl"):
    """Modular save function that accepts a dictionary of game data"""
    try:
        # Clean data before saving
        cleaned_organisms, cleaned_food = clean_game_state(game_data['organisms'], game_data['food_list'])
        
        # Prepare full game state
        full_state = {
            'organisms': cleaned_organisms,
            'food_list': cleaned_food,
            'game_clock': game_data['game_clock'],
            # Add other game state variables here as needed
        }
        
        with open(filename, 'wb') as file:
            pickle.dump(full_state, file)
        print(f"Game state saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving game state: {e}")
        return False

def load_game_state(filename="game_state.pkl"):
    """Modular load function that returns game state dictionary"""
    try:
        with open(filename, 'rb') as file:
            loaded_state = pickle.load(file)
        
        # Clean loaded data
        organisms, food_list = clean_game_state(loaded_state['organisms'], loaded_state['food_list'])
        
        # Return structured game state
        return {
            'organisms': organisms,
            'food_list': food_list,
            'game_clock': loaded_state.get('game_clock', 0.0),
            # Add other game state variables here as needed
        }
    except FileNotFoundError:
        print("No saved game state found.")
        return None
    except Exception as e:
        print(f"Error loading game state: {e}")
        return None