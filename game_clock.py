class GameClock:
    def __init__(self, initial_time=0.0):
        self.total_time = initial_time  # Real seconds (1 sec = 1 game day)
        self.days_per_season = 30

    def update(self, delta_time):
        """Update clock with real-time delta (seconds)"""
        self.total_time += delta_time

    def get_state(self):
        """Return current clock state for saving"""
        return self.total_time

    def set_state(self, saved_time):
        """Restore clock state from saved data"""
        self.total_time = saved_time

    def get_season_day(self):
        """Returns tuple: (season_number 1-4, day_in_season 1-30)"""
        total_days = int(self.total_time)
        season = (total_days // self.days_per_season) % 4 + 1
        day = (total_days % self.days_per_season) + 1
        return season, day

    def get_total_days(self):
        return int(self.total_time)