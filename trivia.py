import os
import time
import spectral.io.envi as e
import numpy as np
from pprint import pprint
from imageprocess import *
from utils import *

if __name__ == '__main__':
    pass
    # root_dir = r'H:\20240401_xinjiang_border'

    # input_dir = r'H:\20240214_ASTER_Group1\PyPreprocessed'
    # work_dir = r'H:\20240214_ASTER_Group1\0422clip'
    #
    # # csv_fn = r"H:\20240312_AHSI_Group1\hyperspectral_group1_rawdata_all.csv"
    # fn_list = get_fn_list(input_dir, ext='hdr', recursive=False)
    #
    # if not os.path.exists(work_dir):
    #     os.makedirs(work_dir)
    #
    # steps = ['MNF']
    #
    # no_exist_list = []
    #
    # for idx, fn in enumerate(fn_list, start=1):
    #     start_time = time.time()
    #
    #     print(f'Processing {get_fn_info(fn).base} | {idx}/{len(fn_list)} ...')
    #     input_raster_uri = fn
    #     # output_raster_uri = input_raster_uri.replace(input_dir, work_dir)
    #     # if not os.path.exists(os.path.dirname(output_raster_uri)):
    #     #     os.makedirs(os.path.dirname(output_raster_uri))
    #
    #     basename = get_fn_info(fn).base
    #     print(input_raster_uri)
    #     opath = create_opath(input_raster_uri, work_dir, steps, simplify_basename=False)
    #
    #     if not os.path.exists(input_raster_uri):
    #         print(f'{input_raster_uri} does not exist')
    #         no_exist_list.append(basename)
    #     else:
    #         envi_raster = ENVIRaster(input_raster_uri)
    #         original_meta = envi_raster.raster.metadata.copy()
    #         acq_time = original_meta['acquisition time'] if 'acquisition time' in original_meta else 'Unknown'
    #         interleave = envi_raster.raster.interleave
    #         nl, ns, nb = envi_raster.raster.shape
    #
    #         MNF_raster = envi_raster.forward_mnf_envi(opath['MNF'], opath['MNF'].replace('.hdr', '.sta'))
    #
    #         output_fn = MNF_raster.fn
    #         output_meta_fn = output_fn.replace(get_fn_info(output_fn).ext, 'hdr')
    #         output_meta = e.open(output_meta_fn).metadata
    #         output_meta['acquisition time'] = original_meta['acquisition time']
    #         output_meta['data ignore value'] = 0.0
    #         # output_meta['sensor type'] = 'ASTER'
    #         # output_meta['data units'] = 'W m^-2 sr^-1 um^-1'
    #         # output_meta['reflectance scale factor'] = 10000.0
    #
    #         e.write_envi_header(output_meta_fn, output_meta)
    #
    #         end_time = time.time()
    #         consume = format_time(end_time - start_time)
    #         print(f"Workflow consumed time: {consume}")
    #
    # pprint(no_exist_list)
