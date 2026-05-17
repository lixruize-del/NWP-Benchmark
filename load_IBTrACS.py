import pandas as pd


class BestTracker:
    def __init__(self, file_path):
        """
        Initialize the TyphoonTracker class and load the specified IBTrACS dataset.

        Args:
            file_path (str): File path of the IBTrACS dataset.
        """
        self.file_path = file_path
        self.data = None
        self._load_data()

    def _load_data(self):
        """
        Load and clean the IBTrACS dataset.
        """
        try:
            # Read the CSV file
            df = pd.read_csv(self.file_path, low_memory=False)
            
            # Convert the SEASON column to numeric, drop invalid values
            df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
            df = df.dropna(subset=['SEASON'])
            df['SEASON'] = df['SEASON'].astype(int)
            
            # Save the cleaned data
            self.data = df
        except Exception as e:
            print(f"Failed to load data: {e}")

    def get_typhoon_path(self, name, season, start_time, end_time, time_step=None):
        """
        Retrieve the path data of a specified typhoon.

        Args:
            name (str): Name of the typhoon.
            season (int): Year of the typhoon.
            start_time (str): Start time in the format 'YYYY-MM-DD HH:mm:ss'.
            end_time (str): End time in the format 'YYYY-MM-DD HH:mm:ss'.
            time_step (int): Time step interval (in hours), optional. If provided, filters data by the time interval.

        Returns:
            pd.DataFrame: DataFrame containing the typhoon path with latitude, longitude, and time information.
        """
        try:
            # Filter data for the specified season and typhoon name
            name_mask = self.data['NAME'].fillna('').astype(str).str.contains(
                name, case=False, regex=False
            )
            typhoon_data = self.data[
                (self.data['SEASON'] == season) & name_mask
            ].copy()
            
            if typhoon_data.empty:
                print(f"No data found for typhoon '{name}' in season {season}.")
                return pd.DataFrame()

            # Check if 'ISO_TIME' exists
            if 'ISO_TIME' not in typhoon_data.columns:
                print("Error: 'ISO_TIME' column is missing from the dataset.")
                return pd.DataFrame()

            # Convert the ISO_TIME column to datetime format
            typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
            typhoon_data = typhoon_data.dropna(subset=['ISO_TIME'])

            # Filter data within the specified time range
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            filtered_data = typhoon_data[
                (typhoon_data['ISO_TIME'] >= start_time)
                & (typhoon_data['ISO_TIME'] <= end_time)
            ].copy()
            
            if filtered_data.empty:
                print(f"No data found for typhoon '{name}' in the specified time range.")
                return pd.DataFrame()

            # Convert latitude and longitude to numeric
            filtered_data['LAT'] = pd.to_numeric(filtered_data['LAT'], errors='coerce')
            filtered_data['LON'] = pd.to_numeric(filtered_data['LON'], errors='coerce')
            filtered_data = filtered_data.dropna(subset=['LAT', 'LON'])
            
            # If time_step is specified, filter data by the time step interval
            if time_step:
                filtered_data = self._filter_by_time_step(filtered_data, start_time, time_step)
                # print(filtered_data)
            return filtered_data[['ISO_TIME', 'LAT', 'LON']]  # Return time, latitude, and longitude
        except Exception as e:
            print(f"Error retrieving typhoon path: {e}")
            return pd.DataFrame()

    def _filter_by_time_step(self, data, start_time, time_step):
        """
        Filter typhoon path data by a time step interval starting from the specified start time.

        Args:
            data (pd.DataFrame): Typhoon path data.
            start_time (pd.Timestamp): Start time.
            time_step (int): Time step interval (in hours).

        Returns:
            pd.DataFrame: Filtered typhoon path data.
        """
        try:
            # Ensure 'ISO_TIME' is a datetime column
            if 'ISO_TIME' not in data.columns:
                raise KeyError("'ISO_TIME' column is missing from the data.")
                       
            # Set 'ISO_TIME' as the index

            data = data.set_index('ISO_TIME')            
            # Generate a time range starting from the specified start time
            time_range = pd.date_range(start=start_time, end=data.index.max(), freq=f'{time_step}h')
            
            # Filter data by matching the generated time range
            resampled_data = data.loc[data.index.intersection(time_range)].reset_index()
            resampled_data.rename(columns={"index": "ISO_TIME"}, inplace=True)
            return resampled_data
        except Exception as e:
            print(f"Error during time step filtering: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    from tc_paths import ibtracs_csv_path

    tracker = BestTracker(ibtracs_csv_path())
    
    # Retrieve typhoon path
    result = tracker.get_typhoon_path(
        name="YINXING",
        season=2024,
        start_time="2024-11-04 00:00:00",
        end_time="2024-11-14 23:59:59",
        time_step=6  # Select data every 6 hours
    )
    
    # Print the result
    print(result)