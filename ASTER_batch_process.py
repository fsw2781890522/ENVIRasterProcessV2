import os
import time
import numpy as np
from pprint import pprint
from imageprocess import *
from utils import *

if __name__ == '__main__':

    root_dir = r'H:\20240508_chen'

    input_dir = os.path.join(root_dir, 'ASTER_original_img')
    work_dir = os.path.join(root_dir, 'ASTER_processing')

    # csv_fn = r"H:\20240411_Group2_ASTER\filename_20240421.csv"
    # fn_list = get_fn_list(input_dir, ext='hdr', csv_fn=csv_fn, csv_header='filename')

    fn_list = get_fn_list(input_dir=input_dir, ext='hdr')

    # fn_list = [r"H:\20240508_chen\AST_L1T_00311262002071051_20150428002432_19898.hdf"]

    # MNF_dir = os.path.join(root_dir, 'MNF')
    # combine_dir = os.path.join(root_dir, 'Py_band_combination')
    # alter_dir = os.path.join(root_dir, 'PyAlteration')
    # PCA_dir = os.path.join(root_dir, 'PCA')

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # if not os.path.exists(MNF_dir):
    #     os.makedirs(MNF_dir)

    false_list = []
    time_list = []
    file_list = []
    no_exist_list = []

    steps = ['rad', ['VNSW_BIL', 'TIR_BIL'], ['fla', 'fla_mod', 'cloud', 'water', 'cor', 'cor_mod'], 'stack', 'exc',
             'pro']

    """
    complete workflow: ['rad', ['VNSW_BIL', 'TIR_BIL'], ['fla', 'fla_mod', 'cloud', 'water', 'cor', 'cor_mod'], 
    'stack', 'exc', 'pro']
    step1 'excision': self.edge_excision()
    step2 'rad': self.radio_cali()
    
    Note: excision and rad have been integrated into a single method: self.radio_cali_edge_exc()
    
    step3 'VNSW_BIL', 'TIR_BIL': self.export_bands()
    step4 'fla', 'cloud', 'water', 'cor': self.flaash_stm_cor(), self.thermal_atm_cor()
    step5 'stack': ENVIRaster.layer_stack()
    step6 'pro': self.edge_excision()
    """

    for idx, fn in enumerate(fn_list, start=1):
        print(f'Processing {get_fn_info(fn).base} | {idx}/{len(fn_list)} ...')
        input_raster_uri = fn
        basename = get_fn_info(fn).base
        opath = create_opath(input_raster_uri, work_dir, steps)
        # output_fn = os.path.join(output_dir, f'{basename}_pro.img')
        # MNF_fn = os.path.join(MNF_dir, f'{basename}_MNF.img')

        # if os.path.exists(MNF_fn):
        #     print(f'{basename} already preprocessed')
        #     continue

        if not os.path.exists(input_raster_uri):
            print(f'{input_raster_uri} does not exist')
            no_exist_list.append(basename)
        else:
            envi_raster = ENVIRaster(input_raster_uri)
            original_meta = envi_raster.raster.metadata
            acq_time = original_meta['acquisition time']
            interleave = envi_raster.raster.interleave
            nl, ns, nb = envi_raster.raster.shape

            if nb < 14:
                print(f'Incomplete Data ({nb} bands)')
                false_list.append(basename)
            else:

                start_time = time.time()

                time_list.append(original_meta['acquisition time'])
                file_list.append(basename)

                raster = envi_raster.radio_cali_edge_exc(opath['rad'], data_ignore_value=0.0, out_interleave='bil',
                                                         vnsw_scale=0.1,
                                                         tir_scale=1,
                                                         dtype=np.float32)

                vnsw = raster \
                    .export_bands(opath['VNSW_BIL'], bands=range(9), out_interleave='bil') \
                    .flaash_atm_cor(opath['fla'], opath['cloud'], opath['water'], acq_time, data_ignore_value=0.0,
                                    atm_model='Tropical Atmosphere', sensor_type='ASTER') \
                    .modify_value(opath['fla_mod'], scale=1, max_val=10000, min_val=0, add=1, data_ignore_value=0.0)

                tir = raster \
                    .export_bands(opath['TIR_BIL'], bands=range(9, 14), out_interleave='bil') \
                    .thermal_atm_cor(opath['cor']) \
                    .modify_value(opath['cor_mod'], scale=1, min_val=0, data_ignore_value=0.0)

                stack = ENVIRaster.layer_stack([vnsw, tir], opath['stack'], sensor_type='ASTER') \
                    .edge_excision(opath['exc']) \
                    .remove_edge(opath['pro'], data_ignore_value=0, width=30)

                output_fn = stack.fn
                output_meta_fn = output_fn.replace(get_fn_info(output_fn).ext, 'hdr')
                output_meta = e.open(output_meta_fn).metadata
                output_meta['acquisition time'] = original_meta['acquisition time']
                output_meta['data ignore value'] = 0.0
                output_meta['sensor type'] = 'ASTER'
                output_meta['unit for thermal infrared bands'] = 'W/m2/μm/sr'
                output_meta['reflectance scale factor'] = 10000.000000

                e.write_envi_header(output_meta_fn, output_meta)

                # if in_list('rad', steps):
                #     raster = raster.radio_cali_edge_exc(opath['rad'], data_ignore_value=0.0, out_interleave='bil',
                #                                         vnsw_scale=0.1,
                #                                         tir_scale=1,
                #                                         dtype=np.float32)
                #
                # if in_list('VNSW_BIL', steps):
                #     vnsw = raster.export_bands(opath['VNSW_BIL'], bands=range(9), out_interleave='bil')
                # if in_list('TIR_BIL', steps):
                #     tir = raster.export_bands(opath['TIR_BIL'], bands=range(9, 14), out_interleave='bil')
                #
                # if in_list('fla', steps):
                #     vnsw = vnsw.FLAASH(opath['fla'], opath['cloud'], opath['water'])
                # if in_list('cor', steps):
                #     tir = tir.TIR_(opath['cor'])
                # # if in_list('fla_mod', steps):
                # #     Preprocess.modifyValue(opath['fla'], opath['fla_mod'], scale=10000, max_val=10000, min_val=0,
                # #                            data_ignore_value=0, dtype=np.int16, time=meta['acquisition time'])
                # # if in_list('cor_mod', steps):
                # #     Preprocess.modifyValue(opath['cor'], opath['cor_mod'], scale=1, min_val=0,
                # #                            data_ignore_value=0, dtype=np.float32, time=meta['acquisition time'])
                #
                # if in_list('stack', steps):
                #     ENVIRaster.layer_stack([vnsw, tir], opath['stack'])
                #
                # if in_list('excision', steps):
                #     Preprocess.ASTER_EdgeExcision(opath['stack'], output_fn)
                #
                #     # if in_list('rmEdge', steps):
                #     #     Task.removeEdge(opath['stack'], opath['rmEdge'], pixels=100)
                #     #     # Task.removeEdge(fn, opath['rmEdge'], pixels=100)
                #     #     Preprocess.ASTER_EdgeExcision(opath['rmEdge'], output_fn)
                #
                #     # Add time attr to metadata (time lost in Task.removeEdge)
                #     output_meta_fn = output_fn.replace(get_fn_info(output_fn).ext, 'hdr')
                #     output_meta = e.open(output_meta_fn).metadata
                #     output_meta['acquisition time'] = meta['acquisition time']
                #     output_meta['sensor type'] = 'ASTER'
                #     # pprint(output_meta)
                #     e.write_envi_header(output_meta_fn, output_meta)
                #
                # if in_list('MNF', steps):
                #     Task.MNF_Trans(output_fn, MNF_fn)
                #
                # if in_list('combine', steps):
                #     features = [
                #         'AlOH_minerals,advanced_argillic_alteration',
                #         'Clay,amphibole,laterite',
                #         'Gossan,alteration,host_rock(1)',
                #         'Decorellation',
                #         'Silica,carbonate',
                #         'Discrimination_for_mapping',
                #         'Discrimination_in_sulphide_rich_areas',
                #         'Silica,Fe2+',
                #         'Enhanced_structual_features']
                #
                #     for feature in features:
                #         BandMath.ASTER_combine(output_fn, MNF_fn, feature, combine_dir)
                #
                # if in_list('alter', steps):
                #     PCA_fn = os.path.join(MNF_dir, f'{basename}_PCA.img')
                #     # alter_dir = os.path.join()
                #     alters = ['Fe_PC3', 'Fe_PC4', 'Al-OH_sericite', 'Al-OH_kaolinite', 'propy']
                #     PCA_bands = ['1234', '1346', '1345', '1348']
                #     alter_path = create_opath(input_raster_uri, alter_dir, steps=alters)
                #     PCA_path = create_opath(input_raster_uri, PCA_dir, steps=PCA_bands)
                #
                #     BandMath.find_alteration(output_fn, PCA_path['1234'], alter_path['Fe_PC3'], bands=[1, 2, 3, 4],
                #                              feature_PC=3)
                #     BandMath.find_alteration(output_fn, PCA_path['1234'], alter_path['Fe_PC4'], bands=[1, 2, 3, 4],
                #                              feature_PC=4)
                #     BandMath.find_alteration(output_fn, PCA_path['1346'], alter_path['Al-OH_sericite'],
                #                              bands=[1, 3, 4, 6],
                #                              feature_PC=4, inverse=True)
                #     BandMath.find_alteration(output_fn, PCA_path['1345'], alter_path['Al-OH_kaolinite'],
                #                              bands=[1, 3, 4, 5],
                #                              feature_PC=4)
                #     BandMath.find_alteration(output_fn, PCA_path['1348'], alter_path['propy'], bands=[1, 3, 4, 8],
                #                              feature_PC=3, inverse=True)

                end_time = time.time()
                consume = format_time(end_time - start_time)
                print(f"Workflow consumed time: {consume}")

    if false_list:
        pprint(false_list)
    if no_exist_list:
        pprint(no_exist_list)
