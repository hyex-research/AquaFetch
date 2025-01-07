
import os
import site   # so that water_datasets directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

if __name__ == "__main__":
    logging.basicConfig(filename='test_lamah.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from water_datasets import LamaHCE, LamaHIce

from utils import (
    test_dataset,
    test_dynamic_data,
    test_attributes
    )

stations = {'D': [859, 859, 454], 'H': [859, 859, 454]}
static = {'D': [80, 81, 80], 'H': [84, 85, 80]}
num_dyn_attrs = {'D': 22, 'H': 16}
len_dyn_data = {'D': 14244, 'H': 341856}
yearly_steps = {'D': 366, 'H': 8761}  # 8784

for idx, dt in enumerate(['total_upstrm', 'diff_upstrm_all', 'diff_upstrm_lowimp_D']):

    logger.info(f'testing for {dt} at daily time step')

    ds_eu = LamaHCE(timestep='D', data_type=dt, path=gscad_path)

    test_dataset(ds_eu,
                 stations['D'][idx],
                    14244,
                    static['D'][idx],
                    num_dyn_attrs=22,
                    test_df=True,
                    yearly_steps=366)


for idx, dt in enumerate(['total_upstrm',
                          'intermediate_all', 'intermediate_lowimp']):

    logger.info(f'testing for {dt} at hourly time step')

    ds_eu = LamaHCE(timestep='H', data_type=dt, path=gscad_path, verbosity=4)

    test_dataset(ds_eu,
                 stations['H'][idx],
                    341856,
                    static['H'][idx],
                    num_dyn_attrs=16,
                    test_df=False,
                    yearly_steps=8761)

##  **** LamaHIce ****
stations = {
    'D': [111, 107,  107],
    'H': [76]
    }
length = {'D': 26298, 'H': 412825}
num_dynamic = {'D': 36, 'H': 28}
yr_steps = {'D': 366, 'H': 8784}

for idx, data_type in enumerate(['total_upstrm', 
                                 #'intermediate_all', 'intermediate_lowimp'
                    ]):
        
    logger.info(f'testing for {data_type}, at hourly timestep')

    dataset = LamaHIce(path=gscad_path, timestep="H", data_type=data_type, verbosity=4)

    test_dataset(dataset, 
                    76, 
                    412825, 
                    138, 
                    num_dyn_attrs=28,
                    yearly_steps=8761
                    )


for idx, data_type in enumerate(['total_upstrm', 
                                 'intermediate_all', 
                                 'intermediate_lowimp'
                    ]):
        
    logger.info(f'testing for {data_type}, at daily timestep')

    dataset = LamaHIce(path=os.path.join(gscad_path, 'LamaHIce_daily'), timestep='D', data_type=data_type)

    test_dataset(dataset, 
                    stations['D'][idx], 
                    26298, 
                    154, 
                    36,
                    yearly_steps=366
                    )
