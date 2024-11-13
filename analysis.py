import configparser as ConfigParser;
import osmnx as ox;
import xarray as xr;
import numpy as np
from shapely.geometry import mapping;
import matplotlib.pyplot as plt;

configParser = ConfigParser.RawConfigParser()
configFilePath = r'C:\Users\ekleinholkenborg\ABW\config.txt'
configParser.read(configFilePath)

# place_name              = configParser.get('country-configs', 'place_name')
# months_of_interest      = configParser.get('country-configs', 'months_of_interest')
# start_forecast          = configParser.get('country-configs', 'start_forecast')
# place_name              = configParser.get('country-configs', 'place_name')
# return_period_obs       = configParser.get('standard-configs', 'return_period_obs')
# return_period_fc        = configParser.get('standard-configs', 'return_period_fc')
# ensemblenr              = configParser.get('standard-configs', 'ensemblenr')
# start_year              = configParser.get('standard-configs', 'start_year')
# end_year                = configParser.get('standard-configs', 'end_year')
# target_resolution       = configParser.get('standard-configs', 'target_resolution')
# variable                = configParser.get('standard-configs', 'variable')
# base_path_obs           = configParser.get('paths', 'base_path_obs')
# base_path_fc            = configParser.get('paths', 'base_path_fc')

# # def open_nc_file(path_to_data):
# #     """
# #     Open a non-ensemble dataset of a region.

# #     Parameters:
# #     - path_to_data (str): Path to the NetCDF file.

# #     Returns:
# #     - nc_file: NetCDF dataset.
# #     """

# #     nc_file = xr.open_dataset(path_to_data)

# #     return nc_file

# # def resample_nc_file(nc_file, target_resolution=0.001):
# #     """
# #     Resample a NetCDF file to match the shapefile resolution.

# #     Parameters:
# #     - nc_file: NetCDF dataset 
# #     - target_resolution (float): Target resolution in degrees. Defaults to 0.001 degrees.

# #     Returns:
# #     - resampled_ncfile: Resampled NetCDF dataset.
# #     """

# #     try: 
# #         nc_file.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    
# #     except:
# #         nc_file.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    
# #     nc_file.rio.write_crs("epsg:4326", inplace=True)
# #     resampled_ncfile = nc_file.rio.reproject(resolution=target_resolution, dst_crs="epsg:4326")

# #     return resampled_ncfile

# def resample_to_country(place_name, path_to_data, target_resolution=None):
#     """
#     Clip data from a NetCDF file to the country shape.

#     Parameters:
#     - gdf_country: The shape of the country in a GeoDataFrame.
#     - nc_file: NetCDF dataset.

#     Returns:
#     - clipped_by_country: NetCDF database clipped by country shape.
#     """
#     nc_file = xr.open_dataset(path_to_data)

#     if target_resolution == None:
#         continue 
#         else:
#             try: 
     
#                 nc_file.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    
#             except:
#                 nc_file.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    
#             nc_file.rio.write_crs("epsg:4326", inplace=True)
#             nc_file = nc_file.rio.reproject(resolution=target_resolution, dst_crs="epsg:4326")

#     gdf_country = ox.geocode_to_gdf(place_name)
#     clipped_by_country = nc_file.rio.clip(gdf_country.geometry.apply(mapping), gdf_country.crs, drop=True)
   
#     return clipped_by_country


# def plot_country_data(clipped_by_country, place_name, position):
#     """
#     Plot clipped data.

#     Parameters:
#     - gdf_country: The shape of the country in a GeoDataFrame.
#     - clipped_by_country: NetCDF database clipped by country shape.
#     - position (int): Position of the data in time dimension to plot.

#     Returns:
#     - fig_country_data: Matplotlib figure object containing the clipped NetCDF data & country shape.
#     """
#     gdf_country = ox.geocode_to_gdf(place_name)
#     fig_country_data, ax = plt.subplots()

#     clipped_by_country[variable].isel(time=position).plot(ax=ax, zorder=-1)  
#     gdf_country.plot(facecolor='none', edgecolor='black', ax=ax)

#     return fig_country_data

# def forecast_data_to_df(base_path, place_name, year, ensemblenr, start_forecast, target_resolution, forecast_type=None, end_year=None):
#     """
#     Extract and aggregate forecast data from NetCDF files for a specific location and time period.
    
#     Parameters:
#     - base_path (str): The base path of the data files.
#     - place_name (str): Name of the place (e.g., "Lesotho").
#     - year (int): The year of the analysis.
#     - ensemblenr (int): If 'forecast' the ensemble nr must be provided.
#     - start_forecast (str): Two digit month nr. of the start of forecast. E.g. September '09'.
    
#     Returns:
#     - time_series_df (pd.DataFrame): The aggregated observation data as a DataFrame, indexed by date.
#     """
#     time_series_df = pd.DataFrame()

#     if forecast_type == None:
#         for r in range(0, ensemblenr+1): 

#             # Construct the file path for the current value of r and year
#             path = base_path.format(r, year, start_forecast)
#             clipped = resample_to_country(place_name, path, target_resolution)
            
#             # Extract precipitation data and calculate mean
#             precipitation_mean = clipped.pr.mean(dim=["lat", "lon"])
        
#             df = pd.DataFrame(precipitation_mean.values, index=precipitation_mean.time)
        
#             # Check if the column already exists in the DataFrame
#             if f'R{r}' in time_series_df.columns:
#                 # If the column exists, update it with new data
#                 time_series_df[f'R{r}'] = pd.concat([time_series_df[f'R{r}'], df], axis=0)
#             else:
#                 # If the column doesn't exist, create it
#                 time_series_df[f'R{r}'] = df
        
#     if forecast_type == "SEAS5":
#         for r in range(0, (end_year+1-year)):  
#             # Construct the file path for the current value of r and year
#             path = base_path.format(year)
#             clipped = resample_to_country(place_name, path, target_resolution)

#             precipitation_mean = clipped.tp.mean(dim=["latitude", "longitude"])
#             df = pd.DataFrame(precipitation_mean.values, index=precipitation_mean.time)

#             # If the column exists, update it with new data
#             time_series_df = pd.concat([time_series_df, df], axis=0)

#             year += 1

#     return time_series_df

# def observation_data_to_df(base_path, place_name, start_year, end_year):
#     """
#     Extract and aggregate observation data from NetCDF files for a specific location and time period.
    
#     Parameters:
#     - base_path (str): The base path of the data files.
#     - place_name (str): Name of the place (e.g., "Lesotho").
#     - start_year (int): The start year of the analysis.
#     - end_year (int): The end year of the analysis.

#     Returns:
#     - time_series_df (pd.DataFrame): The aggregated observation data as a DataFrame, indexed by date.
#     """
    
#     time_series_df = pd.DataFrame()

#     for year in range(start_year, end_year + 1):
#         for month in range(1, 13):
#             # Open the NetCDF file
#             month = "{:02d}".format(month)
#             path = base_path.format(year, month)
#             nc_file = resample_to_country(place_name, path)
            
#             # Extract precipitation data and calculate mean over the entire area
#             precipitation_mean = nc_file.pr.mean(dim=["lat", "lon"])
#             df = pd.DataFrame(precipitation_mean.values, index=precipitation_mean.time, columns=[f'Precipitation'])
            
#             # Concatenate with historical_data DataFrame
#             time_series_df = pd.concat([time_series_df, df], axis=0)
    
#     return time_series_df

# def create_time_series_df(base_path, place_name, start_year, end_year, type, ensemblenr=None, start_forecast=None):
#     """
#     Create a DataFrame of historical data.

#     Parameters:
#     - base_path (str): The base path of the data files.
#     - place_name (str): Name of the place (e.g., "Lesotho").
#     - start_year (int): The start year of the analysis.
#     - end_year (int): The end year of the analysis.
#     - type (str): The type of data to analyze, which 'observation'.
#     - ensemblenr (int): If 'forecast' the ensemble nr must be provided.
#     - start_forecast (str): Two digit month nr. of the start of forecast. E.g. September '09'.

#     Returns:
#         pd.DataFrame: DataFrame containing historical data.
#     """
#     time_series_df = pd.DataFrame()
#     time_series = {}

#     for year in range(start_year, end_year + 1):
#         if type == "observation":
#             time_series = observation_data_to_df(base_path, place_name, start_year, end_year)
            
#         elif type == "forecast":
#             time_series_df = forecast_data_to_df(base_path, place_name, year, ensemblenr, start_forecast) 
#             time_series[str(year)] = xr.Dataset(time_series_df)

#         else:
#             print("Please provide the type of the time series: 'observation' or 'forecast'.")      

#     return time_series

# def time_series_plot(time_series):
#     """
#     Plot daily precipitation over years.

#     Parameters:
#     - time_series (dataframe): Dateframe with timeseries

#     Returns:
#     - fig, ax (matplotlib.figure.Figure, matplotlib.axes.Axes): Matplotlib figure and axes objects.
#     """
#     fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the width and height as needed
#     time_series.plot(ax=ax)

#     # Set y-axis lower limit to 0
#     plt.ylim(bottom=0)

#     # Adding titles and labels
#     plt.title(f'Daily Precipitation Over Years {place_name}')
#     plt.xlabel('Year')
#     plt.ylabel('Precipitation (mm)')
#     plt.grid()
#     plt.legend()

#     return

# def shift_years_for_aggregation(df, months_of_interest):
#     """
#     If season crossess multiple years, shift the year index of the DataFrame for aggregation.

#     Parameters:
#     - df (DataFrame): DataFrame containing precipitation data.
#     - months_of_interest (list): List of months of interest.

#     Returns:
#     - df_pr (DataFrame): DataFrame with shifted year index.
#     """
#     if 1 in months_of_interest:
#         months_in_previous_year = months_of_interest.copy()
#         index_of_one = months_in_previous_year.index(1)
#         del months_in_previous_year[index_of_one:]
        
#         # Shift the year index of all months of the previous year, so we can aggregate them by year
#         for month in months_in_previous_year:
#             df.index = df.index.where(~(df.index.month == month), df.index + pd.offsets.DateOffset(years=1))
    
#     return df

# def aggregate_by_month(shifted_df):
#     """
#     Aggregate precipitation data over years.

#     Parameters:
#     - shifted_df (DataFrame): shifted data over years

#     Returns:
#     - precip_sum (DataFrame): DataFrame with aggregated precipitation sums by year.
#     """
    
#     if type == 'observation':
#         aggregated_by_month_df = shifted_df[shifted_df.index.month.isin(months_of_interest)].resample('A').sum()
#         aggregated_by_month_df.index = aggregated_by_month_df.index.year
#         aggregated_by_month_df.columns = ['Precipitation']

#     if type == 'forecast':
#         aggregated_by_month_df = pd.DataFrame()
#         for year in range(start_year, end_year + 1):  
#             monthly = shifted_df
#             precip_sum = monthly[monthly.index.month.isin(months_of_interest)].sum()
#             precip_sum = pd.DataFrame(precip_sum).T
#             precip_sum = precip_sum.rename_axis(year)
#             precip_sum.index = [year]
#             # Append the sum for the current year to the result DataFrame
#             aggregated_by_month_df = pd.concat([aggregated_by_month_df, precip_sum], ignore_index=False)

#     return aggregated_by_month_df

# def get_terciles(type, aggregated_by_month_df):
#     """
#     Determine the categories for Below Normal Rainfall, Normal Rainfall and Above Normal Rainfall based on the terciles. 

#     Parameters: 
#     - type: 'forecast' or 'observation'
#     - aggregated_by_month_df (pd.DataFrame): The aggregated data by month.

#     Returns:
#     - lower_tercile (int): The value that determines the threshold for Below Normal Rainfall
#     - upper_tercile (int): The value that determines the threshold for Above Normal Rainfall
#     """
#     if type == 'observation':
#         lower_tercile = aggregated_by_month_df.quantile(1/3).values[0]
#         upper_tercile = aggregated_by_month_df.quantile(2/3).values[0]

#     if type == 'forecast': 
#         aggregated_by_month = np.ravel(aggregated_by_month_df)
#         lower_tercile = np.quantile(aggregated_by_month, q=(1/3))
#         upper_tercile = np.quantile(aggregated_by_month, q=(2/3))
    
#     return lower_tercile, upper_tercile

# def get_forecast_event_threshold(return_period, aggregated_by_month_df):
#     aggregated_by_month = np.ravel(aggregated_by_month_df)
#     threshold = np.quantile(aggregated_by_month, q=return_period)

#     return threshold

# def calculate_forecasted_drought_years(threshold, probability_threshold, aggregated_by_month_df):
#     probabilities = pd.DataFrame()

#     for index, row in aggregated_by_month_df.iterrows():
#         # Count values below the threshold
#         probability = (row < threshold).sum()/(len(row))
#         # Append the result to the probabilities DataFrame
#         probabilities.at[index, 'Probability on exceeding threshold in %'] = probability

#     # Determine if drought is forecasted (true/false)
#     if probability_threshold == None:
#         probabilities['Forecasted Drought'] = probabilities['Probability on exceeding threshold in %'] > 0.25
#     else:
#         probabilities['Forecasted Drought'] = probabilities['Probability on exceeding threshold in %'] > probability_threshold

#     return probabilities


# def calculate_observed_drought_years(aggregated_by_month_df, return_period):
#     """
#     Calculate observed drought years based on return period.

#     Args:
#     - return_period (float): The return period for drought threshold calculation.
#     - aggregated_by_month_df (pd.DataFrame): The aggregated data by month.

#     Returns:
#     - event_threshold (pd.Series): The threshold for defining drought events.
#     - drought_years (pd.Index): The years identified as drought years: true/false.
#     """
#     event_threshold = aggregated_by_month_df['Precipitation'].quantile(return_period)
#     aggregated_by_month_df['Observed Drought'] = aggregated_by_month_df['Precipitation'] < event_threshold
#     drought_years = aggregated_by_month_df.index[aggregated_by_month_df["Observed Drought"]] 

#     return event_threshold, aggregated_by_month_df, drought_years

# def create_contingency_table(observed_df, forecasted_df):
#     contingency_table = pd.DataFrame()
#     # contingency_table["Precipitation"] = observed_df["Precipitation"]
#     contingency_table["Observed Drought"] = observed_df["Observed Drought"]
#     contingency_table["Forecasted Drought"] = forecasted_df["Forecasted Drought"]
    
#     # Define conditions
#     conditions = [
#         (contingency_table['Forecasted Drought'] == False) & (contingency_table['Observed Drought'] == True),
#         (contingency_table['Forecasted Drought'] == True) & (contingency_table['Observed Drought'] == True),
#         (contingency_table['Forecasted Drought'] == True) & (contingency_table['Observed Drought'] == False),
#         (contingency_table['Forecasted Drought'] == False) & (contingency_table['Observed Drought'] == False)
#     ]

#     # Define labels for each condition
#     labels = ['Miss', 'Hit', 'False alarm', 'Correct negative']

#     # Add new column based on conditions and labels
#     contingency_table['Categorization'] = np.select(conditions, labels, default='Unknown')

#     tp = sum(contingency_table['Observed Drought'] & contingency_table['Forecasted Drought'])  # Both observed and forecasted as drought
#     fp = sum(~contingency_table['Observed Drought'] & contingency_table['Forecasted Drought'])  # Forecasted as drought, but not observed
#     fn = sum(contingency_table['Observed Drought'] & ~contingency_table['Forecasted Drought'])  # Observed as drought, but not forecasted
#     tn = sum(~contingency_table['Observed Drought'] & ~contingency_table['Forecasted Drought'])  # Neither observed nor forecasted as drought

#     print('tp', tp)
#     print('fp', fp)
#     print('fn', fn)
#     print('tn', tn)

#     pod = tp / (tp + fn) if (tp + fn) != 0 else 0 
#     print('pod', pod)
#     far = fp / (tp + fp) if (tp + fp) != 0 else 0 
#     print('far', far)
#     accuracy = (tp + tn) / (tp + fp + fn + tn)  
#     print('accuracy', accuracy)
#     bias = (tp + fp) / (tp + fn) if (tp + fn) != 0 else 0  # Forecast Bias
#     print('bias', bias)
#     tf = sum(contingency_table['Forecasted Drought']) / len(contingency_table['Forecasted Drought'])  # Trigger Frequency
#     print('tf', tf)
    
#     return contingency_table, pod, far, accuracy, bias, tf


# forecasted_df      = calculate_forecasted_drought_years(base_path_fc, 
#                                                         place_name, 
#                                                         start_year, 
#                                                         end_year, 
#                                                         'forecast', 
#                                                         months_of_interest, 
#                                                         ensemblenr, 
#                                                         start_forecast, 
#                                                         return_period_fc, 
#                                                         probability_threshold=None,
#                                                         aggregated_by_month_df=None)

# observed_df        = calculate_observed_drought_years(base_path_obs, 
#                                                       place_name, 
#                                                       start_year, 
#                                                       end_year, 
#                                                       'observation', 
#                                                       months_of_interest, 
#                                                       return_period_obs, 
#                                                       aggregated_by_month_df=None)

# # create_contingency_table(observed_df, forecasted_df)

print('this is the analysis file')


