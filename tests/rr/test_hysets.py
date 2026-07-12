
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_hysets.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import random

from aqua_fetch import HYSETS

from utils import (
    test_dynamic_data,
    test_static_data,
    test_all_data,
    test_attributes,
    test_fetch_dynamic_features,
    test_fetch_static_feature,
    test_st_en_with_static_and_dynamic,
    test_selected_dynamic_features,
    test_coords,
    test_plot_stations,
    test_area,
    test_q_mm,
    test_boundary,
    test_fetch_dynamic_multiple_stations,
    test_plot_catchment,
)

gscad_path = '/path/to/raw/data'  # replace with actual path


dataset = HYSETS(path=gscad_path, verbosity=5)


def test_canada_stations(dataset):
    """canada_stations() returns the 2375 Watershed_IDs sourced from HYDAT."""
    logging.info(f"test_canada_stations for {dataset.name}")
    stations = dataset.canada_stations()

    assert isinstance(stations, list)
    assert len(stations) == 2375, f"expected 2375 canada stations, got {len(stations)}"
    assert all(isinstance(stn, str) for stn in stations)
    # no duplicate ids
    assert len(set(stations)) == len(stations)
    # every canada station is a valid station of the dataset
    assert set(stations).issubset(set(dataset.stations()))
    return


def test_mexico_stations(dataset):
    """mexico_stations() returns the 46 Watershed_IDs sourced from Mexico."""
    logging.info(f"test_mexico_stations for {dataset.name}")
    stations = dataset.mexico_stations()

    assert isinstance(stations, list)
    assert len(stations) == 46, f"expected 46 mexico stations, got {len(stations)}"
    assert all(isinstance(stn, str) for stn in stations)
    # no duplicate ids
    assert len(set(stations)) == len(stations)
    # every mexico station is a valid station of the dataset
    assert set(stations).issubset(set(dataset.stations()))
    # canada and mexico sources must be disjoint
    assert set(stations).isdisjoint(set(dataset.canada_stations()))
    return


test_canada_stations(dataset)
test_mexico_stations(dataset)

# because it takes very long time, we don't test with all the data
test_dynamic_data(dataset, 0.1, int(14425 * 0.1), 27028)

test_static_data(dataset, None, 14425)
test_static_data(dataset, 0.1, int(14425*0.1))

test_all_data(dataset, 2000, 27028)
test_all_data(dataset, 2000, 27028, True)

test_attributes(dataset, 30, 20, 14425)

test_fetch_dynamic_features(dataset, random.choice(dataset.stations()), 27028)
test_fetch_dynamic_features(dataset, random.choice(dataset.stations()), 27028, True)

test_fetch_dynamic_multiple_stations(dataset, 3, 27028)
test_fetch_dynamic_multiple_stations(dataset, 3, 27028, True)

test_fetch_static_feature(dataset, random.choice(dataset.stations()),
                            14425, 30)

test_st_en_with_static_and_dynamic(dataset, random.choice(dataset.stations()), yearly_steps=366)
test_st_en_with_static_and_dynamic(dataset, random.choice(dataset.stations()), True, yearly_steps=366)

test_selected_dynamic_features(dataset, 27028)
test_selected_dynamic_features(dataset, 27028, True)

test_coords(dataset)

test_plot_stations(dataset)

test_area(dataset)

test_q_mm(dataset)

test_boundary(dataset)

test_plot_catchment(dataset)
