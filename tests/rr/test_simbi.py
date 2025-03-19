
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_simbi.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from aqua_fetch import Simbi

from utils import (
    test_dataset, 
    test_dynamic_data,
    test_attributes,
    test_area,
    test_coords,
    )

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

dataset = Simbi(path=gscad_path)

#test_dataset(dataset, 70, 17167, 232, 3, raise_len_error=False)
    # check that dynamic attribues from all data can be retrieved.
test_dynamic_data(dataset, None, 24, 17167)
test_dynamic_data(dataset, None, 24, 17167, as_dataframe=True)

# check that dynamic data of 10% of stations can be retrieved
test_dynamic_data(dataset, 0.1, 2, 17167, 
                    raise_len_error=False)
test_dynamic_data(dataset, 0.5, int(24*0.5), 17167, True,
                    raise_len_error=False)
test_attributes(dataset, 232, 3, 24)

# dataset.area(dataset.static_data_stations())

test_area(dataset)

test_coords(dataset)
