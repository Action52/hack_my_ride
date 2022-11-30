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
from typing import Optional


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
    dtypes = {'timestamp': str, 'lineID': np.int64, 'directionID':np.int64, 'pointID': np.int64,
              'distancefromPoint': np.float64, 'time': str, 'year':int, 'month':int, 'day':int, 'hour':int, 'minute':int, 'second':int}
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


def get_route_id_from_route_short_name(route_short_name: str, routes_df: pd.DataFrame) -> Optional[str, None]:
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


def get_route_name_from_route_id(route_id: str, routes_df: pd.DataFrame) -> Optional[None,str]:
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


def get_stop_ids_from_route_id(route_id: str, feed: gk.feed.Feed) -> set:
    """
    Retrieves the stop ids associated to a particular route by searching over the programmed trips in gtfs.
    :param route_id: Route id.
    :param feed: Gtfs feed.
    :return:
    """
    stop_ids = feed.stop_times.merge(
        feed.trips.filter(["trip_id", "route_id"])
    ).loc[lambda x: x.route_id.isin([route_id]), "stop_id"].unique()
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


def get_real_headway_by_stop(real_df: pd.DataFrame, stop_id) -> pd.DataFrame:
    """
    Calculates the real headway given a dataframe of real data on a particular stop associated to a route_id.
    :param real_df: Dataframe of real data associated to a particular route.
    :param stop_id: Stop id
    :return:
    """
    vehicles_at_stop = real_df.loc[
        real_df['stop_id'] == stop_id
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


def get_awt(headway_df, stop_id=None, as_str=False) -> Optional[str, dt.timedelta]:
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


def get_line_headways_by_stop(line_id: str, feed: gk.Feed, df_path: str) -> pd.DataFrame:
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
    line_df = retrieve_route_data_at_stops_from_csv(int(route_short_name), df_path, distance_from_point=5.0, feed=feed)
    headways_by_stop = []
    for stop in stops:
        try:
            headway_at_stop = get_real_headway_by_stop(line_df, stop)
            headway_at_stop['mean_headway_seconds'] = get_average_headway(headway_at_stop)
            headways_by_stop.append(headway_at_stop)
        except Exception as e:
            print(e)
    return pd.concat(headways_by_stop)
