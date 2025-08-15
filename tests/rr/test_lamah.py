
import os
import site   # so that aqua_fetch directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

if __name__ == "__main__":
    logging.basicConfig(filename='test_lamah.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from aqua_fetch import LamaHCE, LamaHIce

from utils import test_dataset

stations = [859, 859, 454]
static = [84, 85, 85]

for idx, dt in enumerate(['total_upstrm', 
                          'intermediate_all', 
                          'intermediate_lowimp'
                          ]):

    logger.info(f'testing for {dt} at daily timestep')

    dataset = LamaHCE(timestep='D', data_type=dt, path=os.path.join(gscad_path, 'LamaHCE_daily'), verbosity=4)

    test_dataset(dataset,
                 stations[idx],
                    14244,
                    num_static_attrs=static[idx],
                    num_dyn_attrs=22,
                    yearly_steps=366)

for idx, dt in enumerate(['total_upstrm',
                          'intermediate_all', 
                          'intermediate_lowimp'
                          ]):

    logger.info(f'testing for {dt} at hourly timestep')

    ds_eu = LamaHCE(timestep='H', data_type=dt, path=gscad_path, verbosity=4)

    test_dataset(ds_eu,
                 stations[idx],
                    341856,
                    static[idx],
                    num_dyn_attrs=16,
                    yearly_steps=8761)

##  **** LamaHIce ****

stations = [111, 107, 86]
num_static = [138, 114, 114]

for idx, data_type in enumerate(['total_upstrm', 
                                 'intermediate_all', 
                                 'intermediate_lowimp'
                                 ]):
        
    logger.info(f'testing for {data_type}, at hourly timestep')

    dataset = LamaHIce(path=gscad_path, timestep="H", data_type=data_type, verbosity=4)

    test_dataset(dataset, 
                    num_stations = stations[idx], 
                    dyn_data_len = 412848, 
                    num_static_attrs = num_static[idx], 
                    num_dyn_attrs = 28,
                    yearly_steps = 8761
                    )


num_static = [154, 114, 114]
for idx, data_type in enumerate(['total_upstrm', 
                                 'intermediate_all', 
                                 'intermediate_lowimp'
                                 ]):
        
    logger.info(f'testing for {data_type}, at daily timestep')

    dataset = LamaHIce(path=os.path.join(gscad_path, 'LamaHIce_daily'), 
                       timestep='D', data_type=data_type, 
                       verbosity=4)

    test_dataset(dataset, 
                    stations[idx], 
                    26298, 
                    num_static[idx], 
                    36,
                    yearly_steps=366
                    )
