
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


gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'


dataset = HYSETS(path=gscad_path, verbosity=5)

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
