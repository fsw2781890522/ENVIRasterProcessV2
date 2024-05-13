import os
import time
import numpy as np
from pprint import pprint
from imageprocess import *
from utils import *

if __name__ == '__main__':

    root_dir = r'H:\20240427_Group2_GF5'

    input_dir = os.path.join(root_dir, 'AHSI_original_img')
    work_dir = os.path.join(root_dir, 'AHSI_processing')

    # csv_fn = r"H:\20240312_AHSI_Group1\hyperspectral_group1_rawdata_all.csv"
    fn_list = get_fn_list(input_dir, ext='hdr', recursive=False)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    steps = ['denoise', 'rad', ['fla', 'cloud', 'water', 'fla_mod']]

    no_exist_list = []

    for idx, fn in enumerate(fn_list, start=1):
        start_time = time.time()

        print(f'Processing {get_fn_info(fn).base} | {idx}/{len(fn_list)} ...')
        input_raster_uri = fn

        basename = get_fn_info(fn).base
        print(input_raster_uri)
        opath = create_opath(input_raster_uri, work_dir, steps, simplify_basename=False)

        if not os.path.exists(input_raster_uri):
            print(f'{input_raster_uri} does not exist')
            no_exist_list.append(basename)
        else:
            envi_raster = ENVIRaster(input_raster_uri)
            original_meta = envi_raster.raster.metadata.copy()
            acq_time = original_meta['acquisition time']
            interleave = envi_raster.raster.interleave
            nl, ns, nb = envi_raster.raster.shape

            # month = int(acq_time.split('-')[1])
            # if 7 <= month <= 9:
            #     atm_model = 'Mid-Latitude Summer'
            # else:
            #     atm_model = 'Sub-Arctic Summer'

            pro_raster = envi_raster.denoise(opath['denoise']) \
                .radio_cali(opath['rad'], scale=0.1, out_interleave='bil', data_ignore_value=0.0, dtype=np.float32) \
                .flaash_atm_cor(opath['fla'], opath['cloud'], opath['water'], acq_time, data_ignore_value=0.0,
                                sensor_type='AHSI', atm_model='Tropical Atmosphere', remove_bad_bands=True) \
                .modify_value(opath['fla_mod'], scale=1, max_val=1e4, min_val=0, add=1, data_ignore_value=0.0) \
                # .rpc_ortho(opath['ortho'])

            output_fn = pro_raster.fn
            output_meta_fn = output_fn.replace(get_fn_info(output_fn).ext, 'hdr')
            output_meta = e.open(output_meta_fn).metadata
            output_meta['acquisition time'] = original_meta['acquisition time']
            output_meta['data ignore value'] = 0.0
            output_meta['sensor type'] = 'AHSI'
            output_meta['data units'] = 'W m^-2 sr^-1 um^-1'
            output_meta['reflectance scale factor'] = 10000.0

            e.write_envi_header(output_meta_fn, output_meta)

            end_time = time.time()
            consume = format_time(end_time - start_time)
            print(f"Workflow consumed time: {consume}")

    pprint(no_exist_list)
