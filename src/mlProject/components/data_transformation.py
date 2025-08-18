import os
from src.mlProject import logger
import polars as pl
from sklearn.model_selection import train_test_split
from src.mlProject.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.TransformationConfig = config

    def preprocess_data(self) -> pl.DataFrame:
        """
        Preprocesses Velib bike sharing dataframe by:
        1. Parsing datetime column into date, time and weekday components
        2. Calculating total available bikes and free terminals
        3. Extracting latitude and longitude from station_geo
        
        Args:
            data (pl.DataFrame): Raw Velib dataframe
            
        Returns:
            pl.DataFrame: Preprocessed Velib dataframe with additional columns
        """
        # read data
        df = pl.read_csv(self.TransformationConfig.data_path)

        # filter out non-operative stations
        df = df.filter(pl.col("operative") == True)

        # parse datetime column and extract date and time
        df = df.with_columns(
            # Step 1: Parse the string into a Datetime object
            # Polars' default parser usually handles ISO 8601 format (T separator, Z for UTC)
            datetime = pl.col("datetime").str.to_datetime()
        ).with_columns(
            # Step 2: Extract Date, Time and Weekday from the new Datetime object
            date = pl.col("datetime").dt.date(),
            time = pl.col("datetime").dt.time(),
            weekday = pl.col("datetime").dt.weekday()
        )
        # Extract hour from time column
        df = df.with_columns(
            hour = pl.col("time").cast(str).str.slice(0, 2).cast(pl.Int32)
        )

        # create new columns
        total_available = pl.col("available_mechanical") + pl.col("available_electrical")

        df = df.with_columns(
            total_available = total_available,
            free_terminals = pl.col("capacity") - total_available,
            lat = pl.col("station_geo").str.json_path_match("$.lat").cast(pl.Float64),
            lon = pl.col("station_geo").str.json_path_match("$.lon").cast(pl.Float64)
        )

        # Group by date, hour, and station, taking last value in each hour
        df = df.group_by(["date", "hour", "station_name"]).agg([
            pl.col("weekday").last(),
            pl.col("lat").last(),
            pl.col("lon").last(),
            pl.col("total_available").last(),
            pl.col("available_mechanical").last(), 
            pl.col("available_electrical").last(),
            pl.col("free_terminals").last()
        ])
        
        # drop datetime column and reorder columns
        df = df.select([
            "date",
            "weekday",
            "hour",
            "station_name",
            "total_available",
            "available_mechanical",
            "available_electrical",
            "free_terminals"
        ]).sort(["date", "hour", "station_name"])

        logger.info(f"Preprocessed data. Shape: {df.shape}")

        return df
    

    def create_station_mapping(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Creates a station mapping (station_id <=> station_name) for the dataframe.
        """
        station_mapping = (
            df.sort("station_name")
            .select("station_name")
            .unique()
            .with_row_index("station_id")
        )
        return station_mapping
        


    def create_lagged_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Creates lagged features for the dataframe.
        """
        station_mapping = self.create_station_mapping(df)

        # Add station IDs to original dataframe by joining on station_name
        new_df = df.join(
            station_mapping,
            on="station_name",
            how="left"
        )

        new_df = new_df.with_columns(
            pl.col("station_id").cast(pl.String).cast(pl.Categorical)
        )

        # Replace station_name with station_id and reorder columns
        new_df = new_df.select([
            "station_id",
            "date",
            "hour",
            "total_available",
            "available_mechanical", 
            "available_electrical",
            "free_terminals"
        ])
        
        # Sort by station_id, date and hour to ensure proper lag calculation
        new_df = new_df.sort(["station_id", "date", "hour"])

        # Get unique station IDs
        unique_stations = new_df.get_column("station_id").unique().to_list()

        # Create list to store DataFrames with lagged features for each station
        station_lag_dfs = []

        # For each station, create lagged features
        for station_id in unique_stations:
        # Filter data for current station
            station_df = new_df.filter(pl.col("station_id") == station_id)
            
            # Create lag columns for total_available
            lag_columns = []
            for i in range(1, 25):
                lag_columns.append(
                    pl.col("total_available").shift(i).alias(f"total_available_lag_{i}")
                )
            
            # Add lag columns to station DataFrame
            station_with_lags = station_df.with_columns(lag_columns).select(
                ["station_id", "date", "hour", "total_available"] + 
                [f"total_available_lag_{i}" for i in range(1, 25)]
            )
            
            # Drop rows with any null values (first 24 hours)
            station_with_lags = station_with_lags.drop_nulls()
            
            if len(station_with_lags) > 0:  # Only append if we have data after dropping nulls
                station_lag_dfs.append(station_with_lags)
        
        # Concatenate all station_lag_dfs vertically to form the final df
        lags_df = pl.concat(station_lag_dfs).sort(["date", "hour"])

        logger.info(f"Created lagged features dataframe. Shape: {lags_df.shape}")
        

        return lags_df



    def split_train_test(self, df, split_ratio=0.8):
        # Define the split index
        split_index = int(len(df) * split_ratio)

        # split data temporally
        train = df[:split_index]
        test = df[split_index:]

        # save to csv files annd store them in the given dir
        train.write_csv(os.path.join(self.TransformationConfig.root_dir, "train.csv"))
        test.write_csv(os.path.join(self.TransformationConfig.root_dir, "test.csv"))

        logger.info("Data split into training and test sets")
        print("train.shape", train.shape)
        print("test.shape", test.shape)