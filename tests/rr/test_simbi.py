
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_simbi.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from aqua_fetch import Simbi

from utils import (
    test_dataset as run_shared_tests, 
    test_dynamic_data as check_dynamic_data,
    test_attributes as check_attributes,
    test_area as check_area,
    test_coords as check_coords,
    test_boundary as check_boundary,
    test_plot_catchment as check_plot_catchment,
    test_plot_stations as check_plot_stations,
    test_static_data as check_static_data,
    )

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

dataset = Simbi(path=gscad_path)

#run_shared_tests(dataset, 70, 17167, 232, 3, raise_len_error=False)
    # check that dynamic attribues from all data can be retrieved.
check_dynamic_data(dataset, None, 24, 2800)
check_dynamic_data(dataset, None, 24, 2800, as_dataframe=True)

# check that dynamic data of 10% of stations can be retrieved
check_dynamic_data(dataset, 0.1, 2, 2800, 
                    raise_len_error=False)
check_dynamic_data(dataset, 0.5, int(24*0.5), 2800, True,
                    raise_len_error=False)
check_attributes(dataset, 232, 3, 24)

check_static_data(dataset, 24, 24)

check_area(dataset)

check_coords(dataset)

check_boundary(dataset)

check_plot_catchment(dataset)

check_plot_stations(dataset)
