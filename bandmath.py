import os
import numpy as np
import spectral.io.envi as e
from utils import *


def safe_divide(arr1, arr2):
    return np.divide(arr1, arr2, where=(arr2 != 0), out=np.zeros_like(arr1))


def normalize(arr: np.ndarray, scale=10000):
    print('Normalizing...')

    temp = arr[arr != 0.0]

    offset = np.abs(np.min(temp)) + 1e-13
    # print('offset:', offset)
    temp = temp + offset

    ht, locations = np.histogram(temp, bins=1000)
    ht_acc = np.cumsum(ht) / np.prod(temp.shape)
    '''
    ht_acc: accumulative frequency of histogram
    note: ht, locations = np.histogram(column, bins=200, density=true) easily cause error
    '''
    w1 = np.where(ht_acc >= 0.999)
    w2 = np.where(ht_acc >= 0.001)
    max_val = locations[w1[0][0]]
    min_val = locations[w2[0][0]]
    print(max_val, min_val)

    ltmin = (temp < min_val)
    gtmax = (temp > max_val)

    temp = scale * (temp - min_val) / (max_val - min_val)
    temp[ltmin] = 1e-13
    temp[gtmax] = scale
    arr[arr != 0.0] = temp

    arr[np.isfinite(arr) != 1] = 0.0

    new_arr = arr.copy()

    return new_arr


def rgb_combine(stack_raster_uri, MNF_raster_uri, out_dir, feature_name, scale=10000, dtype=np.int16,
                data_ignore_value=0.0):
    stack_raster_uri = stack_raster_uri.replace(get_fn_info(stack_raster_uri).ext, 'hdr')
    MNF_raster_uri = MNF_raster_uri.replace(get_fn_info(MNF_raster_uri).ext, 'hdr')

    basename = get_fn_info(stack_raster_uri).base
    o_folder = os.path.join(out_dir, feature_name)
    if not os.path.exists(o_folder):
        os.makedirs(o_folder)

    o_fn = os.path.join(o_folder, f'{basename}_{feature_name}.hdr')

    if os.path.exists(o_fn):
        print(f'{o_fn} already generated')
    else:
        stack_raster = e.open(stack_raster_uri)
        MNF_raster = e.open(MNF_raster_uri)
        meta = stack_raster.metadata
        nl, ns, nb = stack_raster.shape

        b1 = np.float32(stack_raster.read_band(0))
        b2 = np.float32(stack_raster.read_band(1))
        b3 = np.float32(stack_raster.read_band(2))
        b4 = np.float32(stack_raster.read_band(3))
        b5 = np.float32(stack_raster.read_band(4))
        b6 = np.float32(stack_raster.read_band(5))
        b7 = np.float32(stack_raster.read_band(6))
        b8 = np.float32(stack_raster.read_band(7))
        b9 = np.float32(stack_raster.read_band(8))
        b10 = np.float32(stack_raster.read_band(9))
        b11 = np.float32(stack_raster.read_band(10))
        b12 = np.float32(stack_raster.read_band(11))
        b13 = np.float32(stack_raster.read_band(12))
        b14 = np.float32(stack_raster.read_band(13))

        MNF_b1 = np.float32(MNF_raster.read_band(0))

        background_locations = (b1 == data_ignore_value)

        R_info, G_info, B_info, refer = '', '', '', ''

        if feature_name == 'vegetation_and_visible_bands':
            R = safe_divide(b3, b2)
            G = b2
            B = b1
            R_info = 'b3 / b2'
            G_info = 'b2'
            B_info = 'b1'
            refer = ''

        if feature_name == 'AlOH_minerals,advanced_argillic_alteration':
            R = safe_divide(b5, b6)
            G = safe_divide(b7, b6)
            B = safe_divide(b7, b5)
            R_info = 'b5 / b6'
            G_info = 'b7 / b6'
            B_info = 'b7 / b5'
            refer = 'Hewson(CSIRO)'

        if feature_name == 'Clay,amphibole,laterite':
            R = safe_divide((b5 * b7), (b6 * b6))
            G = safe_divide(b6, b8)
            B = safe_divide(b4, b5)
            R_info = '(b5 * b7) / (b6 * b6)'
            G_info = 'b6 / b8'
            B_info = 'b4 / b5'
            refer = 'Bierwith'

        if feature_name == 'Gossan,alteration,host_rock(1)':
            R = safe_divide(b4, b2)
            G = safe_divide(b4, b5)
            B = safe_divide(b5, b6)
            R_info = 'b4 / b2'
            G_info = 'b4 / b5'
            B_info = 'b5 / b6'
            refer = 'Volesky'

        if feature_name == 'Gossan,alteration,host_rock(2)':
            R = b6
            G = b2
            B = b1
            R_info = 'b6'
            G_info = 'b2'
            B_info = 'b1'
            refer = ''

        if feature_name == 'Decorellation':
            R = b13
            G = b12
            B = b10
            R_info = 'b13'
            G_info = 'b12'
            B_info = 'b10'
            refer = 'Bierwith'

        if feature_name == 'Silica,carbonate':
            R = safe_divide((b11 ** 2), (b10 * b12))
            G = safe_divide(b13, b14)
            B = safe_divide(b12, b13)
            R_info = '(b11 * b11) / (b10 * b12)'
            G_info = 'b13 / b14'
            B_info = 'b12 / b13'
            refer = 'Nimoyima'

        if feature_name == 'Silica':
            R = safe_divide(b11, b10)
            G = safe_divide(b11, b12)
            B = safe_divide(b13, b10)
            R_info = 'b11 / b10'
            G_info = 'b11 / b12'
            B_info = 'b13 / b10'
            refer = 'CRISO'

        if feature_name == 'Discrimination_for_mapping':
            R = safe_divide(b4, b1)
            G = safe_divide(b3, b1)
            B = safe_divide(b12, b14)
            R_info = 'b4 / b1'
            G_info = 'b3 / b1'
            B_info = 'b12 / b14'
            refer = 'Abdelsalam'

        if feature_name == 'Discrimination_in_sulphide_rich_areas':
            R = b12
            G = b5
            B = b3
            R_info = 'b12'
            G_info = 'b5'
            B_info = 'b3'
            refer = ''

        if feature_name == 'Discrimination(1)':
            R = safe_divide(b4, b7)
            G = safe_divide(b4, b1)
            B = (safe_divide(b2, b3)) * (safe_divide(b4, b3))
            R_info = 'b4 / b7'
            G_info = 'b4 / b1'
            B_info = '(b2 / b3) * (b4 / b3)'
            refer = 'Sultan'

        if feature_name == 'Discrimination(2)':
            R = safe_divide(b4, b7)
            G = safe_divide(b4, b3)
            B = safe_divide(b2, b1)
            R_info = 'b4 / b7'
            G_info = 'b4 / b3'
            B_info = 'b2 / b1'
            refer = 'Abrams(USGS)'

        if feature_name == 'Silica,Fe2+':
            R = safe_divide(b14, b12)
            G = (safe_divide(b1, b2)) * (safe_divide(b5, b3))
            B = MNF_b1
            R_info = 'b14 / b12'
            G_info = '(b1 / b2) * (b5 / b3)'
            B_info = 'MNF b1'
            refer = 'Rowan(USGS)'

        if feature_name == 'Enhanced_structual_features':
            R = b7
            G = b4
            B = b2
            R_info = 'b7'
            G_info = 'b4'
            B_info = 'b2'
            refer = 'Rowan(USGS)'

        R = normalize(R, scale=scale)
        G = normalize(G, scale=scale)
        B = normalize(B, scale=scale)

        R[background_locations] = data_ignore_value
        G[background_locations] = data_ignore_value
        B[background_locations] = data_ignore_value

        idx_data = np.stack([R, G, B], axis=-1)

        idx_meta = dict()
        for key in meta.keys():
            value = meta[key]
            if not isinstance(value, list):
                idx_meta[key] = meta[key]
            elif isinstance(value, list) and len(value) != 14:
                idx_meta[key] = meta[key]

        idx_meta['Feature'] = feature_name
        idx_meta['Red'] = R_info
        idx_meta['Green'] = G_info
        idx_meta['Blue'] = B_info
        idx_meta['Reference'] = refer

        print(f'Writing {feature_name} raster...')
        e.save_image(o_fn, idx_data, metadata=idx_meta, dtype=dtype, interleave='bsq')
