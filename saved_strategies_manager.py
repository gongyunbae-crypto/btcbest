import json
import os
import streamlit as st

class SavedStrategyManager:
    def __init__(self, file_path="saved_strategies.json"):
        self.file_path = file_path
        self.strategies = self.load_strategies()

    def load_strategies(self):
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def save_strategy(self, strategy):
        """
        Saves a strategy to the list if it doesn't already exist (by Description).
        Returns True if saved, False if duplicate.
        """
        # Create a clean copy to save, ensuring no non-serializable objects
        strategy_to_save = strategy.copy()
        
        # Check for duplicates by Description
        for s in self.strategies:
            if s.get('Description') == strategy_to_save.get('Description'):
                return False
        
        self.strategies.append(strategy_to_save)
        self._persist()
        return True

    def delete_strategy(self, index):
        if 0 <= index < len(self.strategies):
            self.strategies.pop(index)
            self._persist()
            return True
        return False

    def delete_all_strategies(self):
        self.strategies = []
        self._persist()
        return True

    def _persist(self):
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.strategies, f, indent=4)
        except IOError as e:
            st.error(f"Failed to save strategies: {e}")

    def get_strategies(self):
        return self.strategies
