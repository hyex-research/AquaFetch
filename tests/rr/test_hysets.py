
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
    test_q_mmd,
    test_boundary,
    test_fetch_dynamic_multiple_stations
)


gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'


def test_hysets():

    hy = HYSETS(path=os.path.join(gscad_path, "HYSETS"), verbosity=5)

    # because it takes very long time, we don't test with all the data
    test_dynamic_data(hy, 0.003, int(14425 * 0.003), 25202)

    test_static_data(hy, None, 14425)
    test_static_data(hy, 0.1, int(14425*0.1))

    test_all_data(hy, 3, 25202)
    test_all_data(hy, 3, 25202, True)

    test_attributes(hy, 28, 5, 14425)

    test_fetch_dynamic_features(hy, random.choice(hy.stations()))
    test_fetch_dynamic_features(hy, random.choice(hy.stations()), True)

    test_fetch_dynamic_multiple_stations(hy, 3,  25202)
    test_fetch_dynamic_multiple_stations(hy, 3, 25202, True)

    test_fetch_static_feature(hy, random.choice(hy.stations()),
                              14425, 28)

    test_st_en_with_static_and_dynamic(hy, random.choice(hy.stations()), yearly_steps=366)
    test_st_en_with_static_and_dynamic(hy, random.choice(hy.stations()), True, yearly_steps=366)

    test_selected_dynamic_features(hy)

    test_coords(hy)

    test_plot_stations(hy)

    test_area(hy)

    test_q_mmd(hy)

    test_boundary(hy)

    return


test_hysets()