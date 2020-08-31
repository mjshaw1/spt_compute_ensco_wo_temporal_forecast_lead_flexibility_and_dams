# -*- coding: utf-8 -*-
"""generate_warning_points.py
    This file containse functions to
    generate GeoJSON warning point
    files based on historical return period data
    and the most recent forecast.
    Created by Alan D. Snow and Scott D. Christensen, 2015-2017.
    License: BSD-3 Clause
"""
# pylint: disable=superfluous-parens, too-many-locals, too-many-statements
from __future__ import unicode_literals

from builtins import str as text
from io import open
from json import dumps
import os

from netCDF4 import Dataset as NETDataset
import numpy as np
import pandas as pd
import xarray
from multiprocessing import Pool
from functools import partial



def geojson_features_to_collection(geojson_features):
    """
    Adds the feature collection wrapper for geojson
    """
    return {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {
                'name': 'EPSG:4326'
            }
        },
        'features': geojson_features
    }


def generate_lsm_warning_points(qout_file, return_period_file, out_directory,
                                threshold):
    """
    Create warning points from return periods and LSM prediction data
    """
    # get the comids in qout file
    with xarray.open_dataset(qout_file) as qout_nc:
        prediction_rivids = qout_nc.rivid.values

    print("Extracting Return Period Data ...")
    return_period_nc = NETDataset(return_period_file, mode="r")
    return_period_rivids = return_period_nc.variables['rivid'][:]
    return_period_20_data = return_period_nc.variables['return_period_20'][:]
    return_period_10_data = return_period_nc.variables['return_period_10'][:]
    return_period_2_data = return_period_nc.variables['return_period_2'][:]
    return_period_lat_data = return_period_nc.variables['lat'][:]
    return_period_lon_data = return_period_nc.variables['lon'][:]
    return_period_nc.close()

    print("Analyzing Forecast Data with Return Periods ...")
    return_20_points_features = []
    return_10_points_features = []
    return_2_points_features = []
    for prediciton_rivid_index, prediction_rivid in\
            enumerate(prediction_rivids):
        # get interim comid index
        return_period_comid_index = \
            np.where(return_period_rivids == prediction_rivid)[0][0]

        # perform analysis on datasets
        return_period_20 = return_period_20_data[return_period_comid_index]
        return_period_10 = return_period_10_data[return_period_comid_index]
        return_period_2 = return_period_2_data[return_period_comid_index]
        lat_coord = return_period_lat_data[return_period_comid_index]
        lon_coord = return_period_lon_data[return_period_comid_index]

        # create graduated thresholds if needed
        if threshold is not None:
            if return_period_20 < threshold:
                return_period_20 = threshold * 10
                return_period_10 = threshold * 5
                return_period_2 = threshold

        # get daily peaks
        with xarray.open_dataset(qout_file) as qout_nc:
            daily_df = \
                qout_nc.isel(rivid=prediciton_rivid_index).Qout\
                       .resample('D', dim='time', how='max', skipna=True)\
                       .to_dataframe().Qout

        # generate warnings
        for peak_time, peak_qout in daily_df.iteritems():
            feature_geojson = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon_coord, lat_coord]
                },
                "properties": {
                    "peak": float("{0:.2f}".format(peak_qout)),
                    "peak_date": peak_time.strftime("%Y-%m-%d"),
                    "rivid": int(prediction_rivid),
                }
            }

            if peak_qout > return_period_20:
                return_20_points_features.append(feature_geojson)
            elif peak_qout > return_period_10:
                return_10_points_features.append(feature_geojson)
            elif peak_qout > return_period_2:
                return_2_points_features.append(feature_geojson)

    print("Writing Output ...")
    with open(os.path.join(out_directory, "return_20_points.geojson"), 'w') \
            as outfile:
        outfile.write(text(dumps(
            geojson_features_to_collection(return_20_points_features))))
    with open(os.path.join(out_directory, "return_10_points.geojson"), 'w') \
            as outfile:
        outfile.write(text(dumps(
            geojson_features_to_collection(return_10_points_features))))
    with open(os.path.join(out_directory, "return_2_points.geojson"), 'w') \
            as outfile:
        outfile.write(text(dumps(
            geojson_features_to_collection(return_2_points_features))))

def compare_return_periods_to_thresholds(return_period_rivids, return_period_20_data, 
                                         return_period_10_data, return_period_2_data, 
                                         return_period_lat_data, return_period_lon_data, 
                                         merged_ds, mean_ds, std_ds, max_ds, 
                                         threshold, rivid_indices, rivid):
    rivid_index = rivid_indices[rivid]
    return_rivid_index = np.where(return_period_rivids == rivid)[0][0]
    return_period_20 = return_period_20_data[return_rivid_index]
    return_period_10 = return_period_10_data[return_rivid_index]
    return_period_2 = return_period_2_data[return_rivid_index]
    lat_coord = return_period_lat_data[return_rivid_index]
    lon_coord = return_period_lon_data[return_rivid_index]

    # create graduated thresholds if needed
    if return_period_20 < threshold:
        return_period = 0
        feature_geojson = "None"
    else:# get mean
        mean_ar = mean_ds.isel(rivid=rivid_index)
        # mean plus std
        std_ar = std_ds.isel(rivid=rivid_index)
        std_upper_ar = (mean_ar + std_ar)
        max_ar = max_ds.isel(rivid=rivid_index)
        # std_upper_ar[std_upper_ar > max_ar] = max_ar
        std_upper_size = len(std_upper_ar)
        for b in range(1, std_upper_size):
            if std_upper_ar[b] > max_ar[b]:
                std_upper_ar[b] = max_ar[b]

        combinded_stats = pd.DataFrame({
            'mean': mean_ar.to_dataframe().Qout,
            'std_upper': std_upper_ar.to_dataframe().Qout
        })

        for peak_info in combinded_stats.itertuples():
            if peak_info.mean >= return_period_20:
                return_period = 20
                feature_geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon_coord, lat_coord]
                    },
                    "properties": {
                        "mean_peak": float("{0:.2f}".format(peak_info.mean)),
                        "peak_date": peak_info.Index.strftime("%Y-%m-%d"),
                        "rivid": int(rivid),
                        "size": 1
                    }
                }
            elif peak_info.std_upper > return_period_20:
                return_period = 20
                feature_geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon_coord, lat_coord]
                    },
                    "properties": {
                        "mean_peak": float("{0:.2f}".format(peak_info.std_upper)),
                        "peak_date": peak_info.Index.strftime("%Y-%m-%d"),
                        "rivid": int(rivid),
                        "size": 1
                    }
                }
            elif peak_info.mean > return_period_10:
                return_period = 10
                feature_geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon_coord, lat_coord]
                    },
                    "properties": {
                        "mean_peak": float("{0:.2f}".format(peak_info.mean)),
                        "peak_date": peak_info.Index.strftime("%Y-%m-%d"),
                        "rivid": int(rivid),
                        "size": 1
                    }
                }
            elif peak_info.std_upper > return_period_10:
                return_period = 10
                feature_geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon_coord, lat_coord]
                    },
                    "properties": {
                        "mean_peak": float("{0:.2f}".format(peak_info.std_upper)),
                        "peak_date": peak_info.Index.strftime("%Y-%m-%d"),
                        "rivid": int(rivid),
                        "size": 1
                    }
                }
            elif peak_info.mean > return_period_2:
                return_period = 2
                feature_geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon_coord, lat_coord]
                    },
                    "properties": {
                        "mean_peak": float("{0:.2f}".format(peak_info.mean)),
                        "peak_date": peak_info.Index.strftime("%Y-%m-%d"),
                        "rivid": int(rivid),
                        "size": 1
                    }
                }
            elif peak_info.std_upper > return_period_2:
                return_period = 2
                feature_geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon_coord, lat_coord]
                    },
                    "properties": {
                        "mean_peak": float("{0:.2f}".format(peak_info.std_upper)),
                        "peak_date": peak_info.Index.strftime("%Y-%m-%d"),
                        "rivid": int(rivid),
                        "size": 1
                    }
                }
            elif peak_info.mean < return_period_2:
                return_period = 0
                feature_geojson = None
            elif peak_info.std_upper < return_period_2:
                return_period = 0
                feature_geojson = None            
    return int(return_period), str(feature_geojson) 

def generate_ecmwf_warning_points(ecmwf_prediction_folder, return_period_file,
                                  out_directory, threshold):
    """
    Create warning points from return periods and ECMWF prediction data
    """

    # get list of prediciton files
    prediction_files = \
        sorted([os.path.join(ecmwf_prediction_folder, f)
                for f in os.listdir(ecmwf_prediction_folder)
                if not os.path.isdir(os.path.join(ecmwf_prediction_folder, f))
                and f.lower().endswith('.nc')])

    ensemble_index_list = []
    qout_datasets = []
    for forecast_nc in prediction_files:
        ensemble_index_list.append(
            int(os.path.basename(forecast_nc)[:-3].split("_")[-1]))
        qout_datasets.append(
            xarray.open_dataset(forecast_nc, autoclose=True).Qout)

    merged_ds = xarray.concat(qout_datasets,
                              pd.Index(ensemble_index_list, name='ensemble'))
    
    # convert to daily max
    #merged_ds = merged_ds.resample('D', dim='time', how='max', skipna=True)
    merged_ds = merged_ds.resample(time="D", skipna=True).max('time')
    # analyze data to get statistic bands
    mean_ds = merged_ds.mean(dim='ensemble')
    std_ds = merged_ds.std(dim='ensemble')
    max_ds = merged_ds.max(dim='ensemble')
    print("Extracting Return Period Data ...")
    return_period_nc = NETDataset(return_period_file, mode="r")
    return_period_rivids = return_period_nc.variables['rivid'][:]
    return_period_20_data = return_period_nc.variables['return_period_20'][:]
    return_period_10_data = return_period_nc.variables['return_period_10'][:]
    return_period_2_data = return_period_nc.variables['return_period_2'][:]
    return_period_lat_data = return_period_nc.variables['lat'][:]
    return_period_lon_data = return_period_nc.variables['lon'][:]
    return_period_nc.close()
    print ("done extracting Return Period Data ...")
    print("Analyzing Forecast Data with Return Periods ...")
    return_points_features = []
    for rivid_index, rivid in enumerate(merged_ds.rivid.values):
        return_rivid_index = np.where(return_period_rivids == rivid)[0][0]
        return_period_20 = return_period_20_data[return_rivid_index]
        return_period_10 = return_period_10_data[return_rivid_index]
        return_period_2 = return_period_2_data[return_rivid_index]
        lat_coord = return_period_lat_data[return_rivid_index]
        lon_coord = return_period_lon_data[return_rivid_index]

        # create graduated thresholds if needed
        if return_period_20 < threshold:
            pass

        else:# get mean
            mean_ar = mean_ds.isel(rivid=rivid_index)
            # mean plus std
            std_ar = std_ds.isel(rivid=rivid_index)
            std_upper_ar = (mean_ar + std_ar)
            max_ar = max_ds.isel(rivid=rivid_index)
            std_upper_size = len(std_upper_ar)
            for a in range(1, std_upper_size):
                if std_upper_ar[a] > max_ar[a]:
                    std_upper_ar[a] = max_ar[a]
 
            combinded_stats = pd.DataFrame({
                'mean': mean_ar.to_dataframe().Qout,
                'std_upper': std_upper_ar.to_dataframe().Qout,
                'std_dev': std_ar.to_dataframe().Qout
            })

            for peak_info \
                    in combinded_stats.itertuples():
                append_flag = False
                if peak_info.mean > return_period_20: 
                    ret_per = 20
                    wp_size = 0
                    ret_per_max = return_period_20
                elif peak_info.std_upper > return_period_20:
                    ret_per = 20
                    wp_size = 1
                    ret_per_max = return_period_20
                elif peak_info.mean > return_period_10:
                    ret_per = 10
                    wp_size = 0
                    ret_per_max = return_period_10
                elif peak_info.std_upper > return_period_10:
                    ret_per = 10
                    wp_size = 1
                    ret_per_max = return_period_10
                elif peak_info.mean > return_period_2:
                    ret_per = 2
                    wp_size = 0
                    ret_per_max = return_period_2
                elif peak_info.std_upper > return_period_2:
                    ret_per = 2
                    wp_size = 1
                    ret_per_max = return_period_2
                else:
                    ret_per = 0
                    wp_size = 9
                    ret_per_max = 0

                feature_geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon_coord, lat_coord]
                    },
                    "properties": {
                        "rivid_rtn_thresh": float("{0:.2f}".format(ret_per_max)),
                        "mean_peak": float("{0:.2f}".format(peak_info.mean)),
                        "std_upper_peak": float("{0:.2f}".format(peak_info.std_upper)),
                        "std_dev": float("{0:.2f}".format(peak_info.std_dev)),
                        "peak_date": peak_info.Index.strftime("%Y-%m-%d"),
                        "return_period": int(ret_per),
                        "rivid": int(rivid),
                        "size": int(wp_size)
                    }
                }

                if peak_info.mean > return_period_20 or peak_info.std_upper > return_period_20:
                    append_flag = True
                elif peak_info.mean > return_period_10 or peak_info.std_upper > return_period_10:
                    append_flag = True
                elif peak_info.mean > return_period_2 or peak_info.std_upper > return_period_2:
                    append_flag = True
                if append_flag:
                    return_points_features.append(feature_geojson)

    print("{0} reaches exceed the 2-, 10-, and 20-Year Return Periods".format(len(return_points_features)))
    print("Writing Output ...")
    if len(return_points_features)>0:
        with open(os.path.join(out_directory, "return_points.geojson"), 'w') \
                as outfile:
            outfile.write(text(dumps(
                geojson_features_to_collection(return_points_features))))
