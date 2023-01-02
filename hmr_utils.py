import os.path

import pandas as pd
import geopandas as gpd
import json
import gtfs_kit as gk
from pathlib import Path
import warnings
import numpy as np
from datetime import datetime
import datetime as dt
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, f1_score
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, Future, wait
#libaries for frequent set detection
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import apriori

warnings.filterwarnings('ignore')


def retrieve_route_data_at_stops_from_csv(route_short_name: int, input_file: str, distance_from_point: int = 5.0,
                                          feed: gk.feed.Feed = None):
    """
    Retrieves a specific route data from the csv.
    :param route_short_name: Name of the bus line. For example 71
    :param input_file: Path to the file that we will be parsing.
    TODO: Modify to allow to work over iterables or over a data folder
    :param distance_from_point: Distance that we will consider as "still around the bus stop".
    :param feed: Gtfs_Kit feed with the data.
    :return:
    """
    stops = feed.stops.copy(deep=True)
    stops['stop_id'] = stops['stop_id'].astype(str)
    dtypes = {'timestamp': str, 'lineID': np.int64, 'directionID': np.int64, 'pointID': np.int64,
              'distancefromPoint': np.float64, 'time': str, 'year': int, 'month':int, 'day':int, 'hour':int, 'minute':int, 'second':int}
    vehicle_position = pd.read_csv(input_file)
    vehicle_position.dropna(inplace=True)
    for col, dtype in dtypes.items():
        vehicle_position[col] = vehicle_position[col].astype(dtype)
    vehicle_position: pd.DataFrame = vehicle_position.loc[
        (vehicle_position['lineID']==route_short_name) &
        (vehicle_position['distancefromPoint'] <= distance_from_point)
    ]
    vehicle_position['time'] =  pd.to_datetime(vehicle_position['time'])
    vehicle_position['pointID'] = vehicle_position['pointID'].astype(str)
    vehicle_position.rename(columns={'pointID': 'stop_id'}, inplace=True)
    vehicle_position = vehicle_position.merge(stops.filter(['stop_id', 'stop_name']))
    return vehicle_position.sort_values(by=['stop_id', 'timestamp','distancefromPoint'])


def get_route_id_from_route_short_name(route_short_name: str, routes_df: pd.DataFrame):
    """
    Since the gtfs data gives a different id to the route name, this method matches the id given a short name.
    Returns the first occurence or None if not found
    For example: '71' maps to '65'
    :param route_short_name: Usual name of the line. I.e: 71
    :param routes_df: Dataframe of gtfs data with the routes information.
    :return:
    """
    routes_matching = routes_df.loc[(routes_df['route_short_name'] == route_short_name)]
    if routes_matching.shape[0] > 0:
        return routes_matching.iloc[0]['route_id']
    return None


def get_route_name_from_route_id(route_id: str, routes_df: pd.DataFrame):
    """
    Retrieves the route name given a route id. Returns the first occurence or None if not found
    For example: '65' maps to '71'
    :param route_id: Route id, if known.
    :param routes_df: Dataframe of gtfs data with the routes information.
    :return:
    """
    routes_matching = routes_df.loc[(routes_df['route_id'] == route_id)]
    if routes_matching.shape[0] > 0:
        return routes_matching.iloc[0]['route_short_name']
    return None


def get_stop_origin_end_from_route_id(route_id: str, routes_df: pd.DataFrame) -> tuple:
    """
    Returns the names of the stops of origin and end of a particular route.
    :param route_id: Id of the route.
    :param routes_df: Dataframe of gtfs data with the routes information.
    :return:
    """
    route_long_name = routes_df.loc[(routes_df['route_id'] == route_id)].reset_index().iloc[0]['route_long_name']
    origin, end = route_long_name.split(" - ")
    return origin, end


def get_stop_ids_from_route_id(route_id: str, feed: gk.feed.Feed, direction_id) -> set:
    """
    Retrieves the stop ids associated to a particular route by searching over the programmed trips in gtfs.
    :param route_id: Route id.
    :param feed: Gtfs feed.
    :return:
    """
    stop_ids = feed.stop_times.merge(
        feed.trips.filter(["trip_id", "route_id", "direction_id"])
    )
    stop_ids = stop_ids.loc[
        (stop_ids.route_id == route_id) & (stop_ids.direction_id==direction_id)
    ]['stop_id'].unique()
    return set(stop_ids)


def get_route_trip_stop_times(route_id, trip_id, feed: gk.feed.Feed) -> pd.DataFrame:
    """
    Retrieves the programmed stop times programmed for a route on a particular trip.
    :param route_id: Route id. For example '71'
    :param trip_id: Trip id. For example '1292371439123'
    :param feed: Feed of gtfs data.
    :return:
    """
    stop_ids = get_stop_ids_from_route_id(route_id, feed)
    trip_stop_times: pd.DataFrame = feed.stop_times.loc[
        (feed.stop_times.stop_id.isin(stop_ids))
    ].merge(
        feed.stops.filter(['stop_id', 'stop_name'])
    ).merge(
        feed.trips.filter(['trip_id', 'route_id', 'service_id'])
    )
    trip_stop_times = trip_stop_times.loc[
        (trip_stop_times.route_id == route_id) &
        (trip_stop_times.trip_id == trip_id)
        ].sort_values(by=['trip_id', 'stop_sequence'])
    trip_stop_times = trip_stop_times.merge(feed.calendar)
    return trip_stop_times


def it_route_trips(route_id, trips_df: pd.DataFrame):
    """
    Iterator that returns the trip_ids associated to a route_id.
    :param route_id: Route id
    :param trips_df: Trips df from gtfs data.
    :return:
    """
    route_trips = trips_df.loc[
        (trips_df.route_id == route_id)
    ]['trip_id'].unique()
    for route_trip in route_trips:
        yield route_trip


def vehicle_position_by_date(vehicle_position_df: pd.DataFrame, from_date: datetime=None,
                             to_date: datetime=None) -> pd.DataFrame:
    """
    Gets the vehicle position between two datetimes.
    :param vehicle_position_df: Dataframe of real data.
    :param from_date: datetime of start
    :param to_date: datetime of end
    :return:
    """
    vehicle_position_dates_df = vehicle_position_df.loc[
        (vehicle_position_df.time >= from_date) &
        (vehicle_position_df.time < to_date)
    ]
    return vehicle_position_dates_df


def get_real_headway_by_stop(real_df: pd.DataFrame, route_id=None, stop_id=None) -> pd.DataFrame:
    """
    Calculates the real headway given a dataframe of real data on a particular stop associated to a route_id.
    :param real_df: Dataframe of real data associated to a particular route.
    :param stop_id: Stop id
    :param route_id: Route id
    :return:
    """
    if route_id is None:
        vehicles_at_stop = real_df.loc[
            real_df['stop_id'] == stop_id
            ]
    else:
        vehicles_at_stop = real_df.loc[
            (real_df['lineID'] == route_id) &
            (real_df['stop_id'] == stop_id)
            ]
    vehicles_at_stop['headway'] = vehicles_at_stop['time'] - vehicles_at_stop['time'].shift(1)
    return vehicles_at_stop


def get_average_headway(headway_df: pd.DataFrame) -> dt.timedelta:
    """
    Gets the average headway provided a dataframe with headway information.
    :param headway_df: Dataframe with headways
    :return:
    """
    return headway_df.headway.mean().total_seconds()


def get_awt(headway_df, stop_id=None, as_str=False):
    """
    :param headway_df: Dataframe with headways
    :param stop_id: Stop id.
    :param as_str: returns the given timedelta as string (useful in case you want to print the output)
    :return:
    """
    if stop_id is not None:
        headway_df = headway_df.loc[headway_df['stop_id'] == stop_id]
    seconds = headway_df.headway.dt.total_seconds()
    awt = (seconds * seconds).sum() / (2 * seconds.sum())
    if as_str:
        awt = str(dt.timedelta(seconds=awt))
    return awt


def get_line_headways_by_stop(line_id: str, feed: gk.Feed, df_path: str = None, line_df=None) -> pd.DataFrame:
    """
    Returns the line headways by stop.
    :param line_id: Line id of the route. For example '65' for line 71.
    :param feed: Gtfs data feed.
    :param df_path: Path from which to read the data and parse the info.
    TODO: Might have to add option to receive existing dataframe.
    :return:
    """
    stops = get_stop_ids_from_route_id(route_id=line_id, feed=feed)
    route_short_name = get_route_name_from_route_id(line_id, feed.routes)
    if df_path is not None:
        line_df = retrieve_route_data_at_stops_from_csv(
            route_short_name=int(route_short_name),
            input_file=df_path,
            distance_from_point=5,
            feed=feed
        )
    headways_by_stop = []
    for stop in stops:
        try:
            headway_at_stop = get_real_headway_by_stop(line_df, stop)
            headway_at_stop['mean_headway_seconds'] = get_average_headway(headway_at_stop)
            headways_by_stop.append(headway_at_stop)
        except Exception as e:
            print(e)
    return pd.concat(headways_by_stop)


def pick_trips_from_route(feed: gk.Feed, route_id='65') -> pd.DataFrame:
    trips = feed.trips.loc[feed.trips.route_id == route_id].merge(
        feed.calendar[['service_id', 'start_date', 'end_date']]).merge(feed.stop_times).merge(
        feed.stops[['stop_id', 'stop_name']])
    trips.drop(columns=['trip_headsign'], inplace=True, headway_split_time=660.0)
    return trips


def retrieve_headways_from_route_at_stop(feed, route_id='65', stop_id='3558', dates: list = ['20210903'],
                                         start_hour='05:00:00', end_hour='23:00:00', moving_average_period=5,
                                         return_by_periods=False, group_by_period='1H', headway_split_time=720.0,
                                         direction_id=0
                                         ) -> pd.DataFrame:
    # For the example let's pick the ones associated to the Stop Cimetiere D'Ixelles
    route_timetable = gk.routes.build_route_timetable(feed, route_id=route_id, dates=dates)
    route_timetable = route_timetable.loc[
        (route_timetable.stop_id == stop_id) & (route_timetable.direction_id == direction_id)].merge(
        feed.calendar).sort_values(by='arrival_time')
    # Now calculate headway
    route_timetable['headway'] = (pd.to_datetime(route_timetable['departure_time'], errors='coerce') - (
        pd.to_datetime(route_timetable['departure_time'], errors='coerce').shift(1))).dt.total_seconds()
    route_timetable: pd.DataFrame = route_timetable.loc[
        (route_timetable['departure_time'] >= start_hour) & (route_timetable['departure_time'] < end_hour)]
    route_timetable['mvavg'] = route_timetable['headway'].rolling(moving_average_period).mean()
    route_timetable['qas_indi_pred'] = None
    route_timetable['qas_indi_real'] = None
    route_timetable.loc[route_timetable['headway'] < headway_split_time, 'qas_indi_real'] = 'REGULARITY'
    route_timetable.loc[route_timetable['headway'] >= headway_split_time, 'qas_indi_real'] = 'PUNCTUALITY'
    route_timetable.loc[route_timetable['mvavg'] < headway_split_time, 'qas_indi_pred'] = 'REGULARITY'
    route_timetable.loc[route_timetable['mvavg'] >= headway_split_time, 'qas_indi_pred'] = 'PUNCTUALITY'
    route_timetable = route_timetable.set_index('departure_time', drop=False)
    route_timetable.rename(columns={'departure_time': 'departure_time_col'}, inplace=True)
    route_timetable.index = pd.to_datetime(route_timetable.index)
    if return_by_periods:
        periods = []
        route_timetables_per_period = route_timetable.groupby(pd.Grouper(freq=group_by_period))  # .mean()
        for idx, route_timetable_per_period in route_timetables_per_period:
            route_timetable_per_period['period'] = idx
            periods.append(route_timetable_per_period)
        # route_timetable_per_period['qas_indi_pred'] = None
        # route_timetable_per_period['qas_indi_real'] = None
        # route_timetable_per_period.loc[route_timetable_per_period['headway'] < headway_split_time, 'qas_indi_real'] = 'REGULARITY'
        # route_timetable_per_period.loc[route_timetable_per_period['headway'] >= headway_split_time, 'qas_indi_real'] = 'PUNCTUALITY'
        # route_timetable_per_period.loc[route_timetable_per_period['mvavg'] < headway_split_time, 'qas_indi_pred'] = 'REGULARITY'
        # route_timetable_per_period.loc[route_timetable_per_period['mvavg'] >= headway_split_time, 'qas_indi_pred'] = 'PUNCTUALITY'
        # route_timetable_per_period.dropna(inplace=True)
        if not all(period is None for period in periods):
            return pd.concat(periods)
        else:
            logging.debug(f"{route_id}, {stop_id}, {periods} no results.")
            return None
    route_timetable.dropna(inplace=True)
    return route_timetable


def retrieve_average_time_periods_from_route(route_timetable_overall, group_by_period='30Min',
                                             headway_split_time=660.0):
    route_timetable_per_period = route_timetable_overall.groupby(pd.Grouper(freq=group_by_period)).mean()
    route_timetable_per_period['qas_indi_pred'] = None
    route_timetable_per_period['qas_indi_real'] = None
    route_timetable_per_period.loc[
        route_timetable_per_period['headway'] < headway_split_time, 'qas_indi_real'] = 'REGULARITY'
    route_timetable_per_period.loc[
        route_timetable_per_period['headway'] >= headway_split_time, 'qas_indi_real'] = 'PUNCTUALITY'
    route_timetable_per_period.loc[
        route_timetable_per_period['mvavg'] < headway_split_time, 'qas_indi_pred'] = 'REGULARITY'
    route_timetable_per_period.loc[
        route_timetable_per_period['mvavg'] >= headway_split_time, 'qas_indi_pred'] = 'PUNCTUALITY'
    if 'qas_rm_forest' in route_timetable_overall.columns:
        route_timetable_per_period['qas_rm_forest_pred'] = route_timetable_overall.groupby(
            pd.Grouper(freq=group_by_period))['qas_rm_forest'].apply(lambda x: x.mode().iloc[0])
    route_timetable_per_period.dropna(inplace=True)
    return route_timetable_per_period


def compute_metrics_qas(route_timetable_at_stop):
    precision = precision_score(route_timetable_at_stop['qas_indi_real'], route_timetable_at_stop['qas_indi_pred'],
                                pos_label='REGULARITY')
    recall = recall_score(route_timetable_at_stop['qas_indi_real'], route_timetable_at_stop['qas_indi_pred'],
                          pos_label='REGULARITY')
    accuracy = accuracy_score(route_timetable_at_stop['qas_indi_real'], route_timetable_at_stop['qas_indi_pred'])
    f1 = f1_score(route_timetable_at_stop['qas_indi_real'], route_timetable_at_stop['qas_indi_pred'],
                  pos_label='REGULARITY')
    return {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1_score': f1}


def plot_headways_from_route_at_stop(route_timetable_at_stop, bars='qas_indi_real'):
    plt.subplots(figsize=(20, 6))
    plt.xticks(rotation=90)
    ax = sns.barplot(x=route_timetable_at_stop.index, y='headway', data=route_timetable_at_stop, hue=bars,
                     palette={'PUNCTUALITY': 'r', 'REGULARITY': 'b'}, alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(ax.get_xticks(), route_timetable_at_stop['mvavg'], color='black')
    plt.plot()


def plot_average_timetable_for_route(route_timetable_per_period, bars='qas_indi_real'):
    plt.subplots(figsize=(20, 6))
    plt.xticks(rotation=90)
    ax = sns.barplot(x=route_timetable_per_period.index, y='headway', data=route_timetable_per_period, hue=bars,
                     palette={'PUNCTUALITY': 'r', 'REGULARITY': 'b'}, alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(ax.get_xticks(), route_timetable_per_period['mvavg'], color='black')
    plt.plot()


def plot_confussion_matrix_predicted_qas_metric(route_timetable_at_stop):
    cm = confusion_matrix(route_timetable_at_stop['qas_indi_real'], route_timetable_at_stop['qas_indi_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def retrieve_headways_from_route(feed, route_id='65', dates: list = ['20210823'],
                                 start_hour='02:00:00', end_hour='23:00:00',
                                 moving_average_period=5,
                                 return_by_periods=False, group_by_period='30Min', headway_split_time=660.0,
                                 direction_id=0, show_tqdm=False) -> pd.DataFrame:
    stop_ids = get_stop_ids_from_route_id(route_id, feed, direction_id)
    headways_from_route = []
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                retrieve_headways_from_route_at_stop, feed, route_id, stop_id,
                dates, start_hour, end_hour, moving_average_period, return_by_periods,
                group_by_period, headway_split_time, direction_id)
            for stop_id in stop_ids]
        if not show_tqdm:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        else:
            bar = tqdm(total=len(stop_ids))
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                bar.update(1)
            bar.close()
    if not all(result is None for result in results):
        return pd.concat(results)
    else:
        logging.debug("All headways from route were None. Skipping.")
        return None


def tag_routes(feed: gk.feed, route_short_names: list = None, dates=['20210903'],
               start_hour='02:00:00', end_hour='23:59:00', moving_average_period=5,
               return_by_periods=True, group_by_period='30Min', headway_split_time=660.0, write_folder="",
               multiprocess=False):
    if route_short_names is None:
        route_ids = [get_route_id_from_route_short_name(route_short_name, feed.routes)
                     for route_short_name in feed.routes.route_short_name.unique()]
    else:
        route_ids = [get_route_id_from_route_short_name(route_short_name, feed.routes)
                     for route_short_name in route_short_names]
    results = []
    if multiprocess:
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for date in dates:
                for route_id in route_ids:
                    for direction_id in [0, 1]:
                        futures.append(executor.submit(
                            retrieve_headways_from_route, feed, route_id, [date], start_hour, end_hour,
                            moving_average_period, return_by_periods, group_by_period,
                            headway_split_time, direction_id, False
                        ))
            for future in tqdm(as_completed(futures), total=len(dates)*len(route_ids)*2):
                result = future.result()
                result.to_parquet(f'{write_folder}tagged_{date}_{route_id}', engine='fastparquet', compression='gzip', index=True,
                                  partition_cols=['date', 'direction_id', 'stop_id'])
    else:
        bar = tqdm(total=(len(dates)*len(route_ids)*2))
        for date in dates:
            for route_id in route_ids:
                for direction_id in [0, 1]:
                    try:
                        result = retrieve_headways_from_route(
                            feed, route_id, [date], start_hour, end_hour,
                            moving_average_period, return_by_periods, group_by_period,
                            headway_split_time, direction_id, False
                        )
                        if result is not None:
                            name = f'{write_folder}tagged_{date}_{route_id}_{direction_id}'
                            if not os.path.isfile(name):
                                result.to_parquet(name, engine='fastparquet',
                                                  compression='gzip', index=True,
                                                  partition_cols=['date', 'direction_id', 'stop_id'])
                            else:
                                result.to_parquet(name, engine='fastparquet',
                                                  compression='gzip', index=True,
                                                  partition_cols=['date', 'direction_id', 'stop_id'], append=True)
                            print(f"Finished processing {route_id}, {date}, {direction_id}")
                        else:
                            logging.debug(f"{route_id}, {date}, {direction_id} returned a null dataframe for all stops.")
                    except Exception as e:
                        logging.debug(f"Error processing {route_id}, {date}, {direction_id}")
                        logging.debug(e)
                    bar.update(1)
        bar.close()

def getBottleneckSets(testset, n=5):
    """
    Gets from incoming stop list, the stops where the delay was bigger than n
    """
    testset['prev'] = testset['stop_name'].shift(1)
    testset['prev_delay'] = testset['delay'].shift(1)
    testset['dif'] = testset['delay'] - testset['prev_delay'] 
    
    subsets=testset.loc[(testset.dif >= 5)][['stop_name', 'prev']] #filter stops with >5 difference in delay
    sets=subsets.values.tolist() #Parse from dataframe to list
    return sets
  
  def getFrequentSets(dataset, min_support=0.1):
    """
    Gets from list of sets, the most frequent
    """
    tr = TransactionEncoder()
    tr_arr = tr.fit(dataset).transform(dataset)
    df = pd.DataFrame(tr_arr, columns=tr.columns_)

    frequent_itemsets = apriori(df, min_support, use_colnames = True)
    return frequent_itemsets

def main():
    logging.basicConfig(level=logging.DEBUG, filename="logging.log")
    data_folder = '/Users/alfredo.leon/Desktop/DataMiningProject/Project Data-20221021/'
    path = Path(data_folder + "gtfs3Sept.zip")
    feed = (gk.read_feed(path, dist_units='km'))
    feed.validate()
    tagged_folder = '/Users/alfredo.leon/Desktop/DataMiningProject/tagged_correct/'
    dates = ['20210908', '20210915', '20210911', '20210918']
    tag_routes(feed, route_short_names=['71', '95', '34', '57'],
               dates=dates,
               write_folder=tagged_folder)


if __name__ == '__main__':
    main()
