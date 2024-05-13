import os
from osgeo import gdal
import spectral
import spectral.io.envi as e
import numpy as np
# from numba import jit
# import cupy as cp
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import envipyengine
from envipyengine import Engine
from auxfunc import *
from algorithms import *
from pprint import pprint

envi = Engine('ENVI')
envipyengine.config.set('engine', r"D:\SOFTWARE\ENVI60\ENVI60\IDL90\bin\bin.x86_64\taskengine.exe")

# IDE won't prompt parameters messages when coding with this decorator's being applied
# because it returns function with *args, **kwargs
"""
def skip(func):
    def warpper(self, output_raster_uri, *args, **kwargs):
        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
            return ENVIRaster(output_raster_uri)
        else:
            return func(self, output_raster_uri, *args, **kwargs)

    return warpper
"""


class ENVITime:
    def __init__(self, acquisition_str):
        """
        FLAASH Task can only recognize 'ENVITime', a customized object in ENVI + IDL, rather than a string.
        Based on information dehydrated 'ENVITime' object in IDL shows,
        Here we construct an 'ENVITime' dict to imitate such an object.

        usage: var = ENVITime(acquisition_time).info

        :param acquisition_str: string of acquisition time in metadata
        """
        # 将 ACQUISITION 字符串转换为 datetime 对象
        acquisition_datetime = datetime.fromisoformat((acquisition_str.split('.')[0]).replace('Z', ''))  # 去除末尾的毫秒、 'Z'

        # 将 datetime 对象转换为 UTC 时间
        acquisition_utc = acquisition_datetime.replace(tzinfo=timezone.utc)

        # 计算 UNIX_MILLISECONDS
        unix_milliseconds = int(acquisition_utc.timestamp() * 1000)

        # 计算 UNIX_SECONDS
        unix_seconds = int(acquisition_utc.timestamp())

        self.info = dict(
            acquisition=acquisition_str,
            unix_milliseconds=unix_milliseconds,
            unix_seconds=unix_seconds,
            factory='Time'
        )


class ENVIRaster:
    """
    This class is essentially an outer shell covering SPyFile class, for adding processing methods.
    Didn't define this class as a subclass,
    because here we defined 'read_band', 'read_bands' and 'load' method for this class,
    however the same name methods exist in SPyFile class, and we defined 'read_band' method
    with 'read_band' in SPyFile class' being called, like:

        def read_band(args): -------------------- self-defined method of this class
            call read_band(args) ---------------- method from SPyFile class

    The methods inherited from SPyFile class will be overridden
    if we create a subclass and redefine methods with the same names.

    Not elegant enough, absolutely.

    usage: var = ENVIRaster(filename)
    """

    def __init__(self, fn):
        """
        :param fn: absolute path string
        """
        # super(ENVIRaster, self).__init__(fn)
        self.fn = fn
        try:
            self.raster = e.open(fn)
        except Exception as exc:
            try:
                self.fn = self.fn.replace('.img', '.hdr')
                self.raster = e.open(self.fn)
            except Exception as exc:
                raise FileNotFoundError(
                    f'Failed with ENVIRaster initialization.\nPlease confirm whether {self.fn} exists or is valid')

    def __getitem__(self, band):
        """
        use 'band_data = raster[band_idx]' to get data (2D-array) of selected band
        """
        return self.read_band(band)

    def read_band(self, band=0):
        """
        When we load a FLAASH result raster using SPy methods,
        the values will be automatically divided by 10000.0 without user's authorization,
        and converted to float type, occupying more memory,
        because of the field's being detected, named 'reflectance scale factor', in metadata.

        Here we let band data multiply the scale factor to maintain their original values,
        and keep their original data type

        :param band: integer band index
        :return: 2D array with shape (nRows, nColumns)
        """

        if band < 0 or isinstance(band, float):
            raise VariableError('band')

        raster = self.raster
        meta = raster.metadata
        dtype = raster.dtype
        nl, ns, nb = raster.shape
        ref_scale = float(meta['reflectance scale factor']) if 'reflectance scale factor' in meta else 1.0
        ref_scale = int(ref_scale) if ref_scale.is_integer() else ref_scale

        return (raster.read_band(band).reshape((nl, ns)) * ref_scale).astype(dtype)

    def read_bands(self, bands):
        """
        read multi-bands data via optimized read_band() method as a 3D array

        :param bands: list or range
        :return: 3D array with shape (nRows, nColumns, nBands) no matter what interleave the original image is
        """
        return np.stack([self.read_band(band) for band in bands], axis=2)

    def load(self):
        """
        read all bands data via optimized read_band() method to avoid memory allocation error

        :return: 3D array with shape (nRows, nColumns, nBands) no matter what interleave the original image is
        """
        nl, ns, nb = self.raster.shape
        return self.read_bands(range(nb))

    def radio_cali(self, output_raster_uri, scale=0.1, out_interleave='bil', data_ignore_value=0.0, dtype=np.float32):
        """
        :param output_raster_uri:
        :param scale:
        :param out_interleave:
        :param data_ignore_value:
        :param dtype:
        :return:
        """

        print('Applying radiometric calibration...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            interleave = meta['interleave']
            nl, ns, nb = raster.shape
            gain = meta['data gain values']
            offset = meta['data offset values']

            new_meta = meta.copy()
            new_data = np.zeros((nl, ns, nb), dtype=np.float32)
            for i in tqdm(range(nb), ncols=100, colour='white'):
                band_data = self.read_band(i)
                temp = scale * (np.float32(gain[i]) * np.float32(band_data) + np.float32(offset[i]))
                temp[band_data == data_ignore_value] = data_ignore_value
                new_data[:, :, i] = temp

            new_meta['data ignore value'] = data_ignore_value
            new_meta.pop('data gain values')
            new_meta.pop('data offset values')

            print(f'Saving as {out_interleave}...')
            e.save_image(output_raster_uri, new_data, metadata=new_meta, dtype=dtype, interleave=out_interleave)
            print('Done')

        return ENVIRaster(output_raster_uri)

    def flaash_atm_cor(self, output_raster_uri, cloud_raster_uri='', water_raster_uri='', acq_time='',
                       data_ignore_value=0.0, atm_model='Tropical Atmosphere',
                       sensor_type='', remove_bad_bands=False):
        """
        :param atm_model:
        :param output_raster_uri:
        :param cloud_raster_uri:
        :param water_raster_uri:
        :param acq_time:
        :param data_ignore_value:
        :param sensor_type:
        :param remove_bad_bands:
        :return:
        """

        print('Applying FLAASH atmospheric correction...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            meta = self.raster.metadata

            task = envi.task('FLAASH')
            input_raster = dict(url=self.fn.replace('.hdr', '.img'), factory='URLRaster')
            parameters = dict(
                INPUT_RASTER=input_raster,
                SENSOR_TYPE='Unknown' if sensor_type == 'AHSI' else 'Multispectral',
                INPUT_SCALE=1,
                OUTPUT_SCALE=10000,
                CALIBRATION_FORMAT='FLAASH',
                SENSOR_ALTITUDE=705.0 if sensor_type == 'ASTER' else float(meta['sensor altitude']),
                DATE_TIME=ENVITime(acq_time).info,
                MODTRAN_ATM=atm_model,
                USE_AEROSOL='Automatic Selection',
                AEROSOL_SCALE_HT=2,
                AER_REFERENCE_VALUE=0,
                CLOUD_RASTER_URI=cloud_raster_uri.replace('.hdr', '.img'),
                WATER_RASTER_URI=water_raster_uri.replace('.hdr', '.img'),
                OUTPUT_RASTER_URI=output_raster_uri.replace('.hdr', '.img')
            )
            if sensor_type == 'AHSI':
                parameters['SOLAR_AZIMUTH'] = float(meta['solar azimuth'])
                parameters['SOLAR_ZENITH'] = float(meta['solar zenith'])

            task.execute(parameters)

            output_meta_fn = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')
            output_meta = e.open(output_meta_fn).metadata
            output_meta['acquisition time'] = acq_time
            output_meta['sensor type'] = sensor_type
            output_meta['data ignore value'] = data_ignore_value
            e.write_envi_header(output_meta_fn, output_meta)

            if remove_bad_bands:
                print('Removing bad bands...')
                fla_raster = ENVIRaster(output_raster_uri)
                # fla_raster = ENVIRaster(self.fn)
                bbl = fla_raster.raster.metadata['bbl']
                bbl = [idx for idx, value in enumerate(bbl) if value == 1]
                new_raster = fla_raster.export_bands(output_raster_uri.replace('_fla.', '_fla2.'), bands=bbl)
                new_fn = new_raster.fn

                fla_raster = None
                new_raster = None

                os.remove(output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'img'))
                os.remove(output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr'))

                os.rename(new_fn.replace(get_fn_info(new_fn).ext, 'img'),
                          new_fn.replace(get_fn_info(new_fn).ext, 'img').replace('_fla2.', '_fla.'))
                os.rename(new_fn.replace(get_fn_info(new_fn).ext, 'hdr'),
                          new_fn.replace(get_fn_info(new_fn).ext, 'hdr').replace('_fla2.', '_fla.'))

                print('Done')

        return ENVIRaster(output_raster_uri)
        # return ENVIRaster(new_fn)

    def remove_bad_bands(self, output_raster_uri):
        """
        :param output_raster_uri:
        :return:
        """
        print('Removing bad bands...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            meta = self.raster.metadata
            bbl = meta['bbl']
            bbl = [idx for idx, value in enumerate(bbl) if value == 1]
            new_raster = self.export_bands(output_raster_uri, bands=bbl)

        return ENVIRaster(output_raster_uri)

    def rpc_ortho(self, output_raster_uri, dem_fn=None, data_ignore_value=0.0):

        print('Applying RPC orthorectification...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            if dem_fn is None:
                dem_fn = r"D:\SOFTWARE\ENVI60\ENVI60\data\GMTED2010.jp2"
            task = envi.task('RPCOrthorectification')
            input_raster = dict(url=self.fn.replace('.hdr', '.img'), factory='URLRaster')
            dem_raster = dict(url=dem_fn, factory='URLRaster')
            parameters = dict(
                INPUT_RASTER=input_raster,
                DEM_RASTER=dem_raster,
                RESAMPLING='Nearest Neighbor',
                OUTPUT_RASTER_URI=output_raster_uri.replace('.hdr', '.img')
            )
            task.execute(parameters)

            output_meta_fn = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')
            output_meta = e.open(output_meta_fn).metadata
            output_meta['data ignore value'] = data_ignore_value
            e.write_envi_header(output_meta_fn, output_meta)

            print('Done')

        return ENVIRaster(output_raster_uri)

    def export_bands(self, output_raster_uri, out_interleave=None, bands=None, dtype=None):

        print('Exporting selected bands...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            nl, ns, nb = raster.shape

            if bands is None:
                bands = range(nb)

            if out_interleave is None:
                out_interleave = meta['interleave']

            if dtype is None:
                dtype = raster.dtype

            print('Generating metadata...')
            new_meta = dict()
            for key in meta.keys():
                value = meta[key]
                if isinstance(value, list) and len(value) == nb:
                    new_meta[key] = [value[i] for i in bands]
                else:
                    new_meta[key] = value
            new_meta['data ignore value'] = 0.0

            new_data = self.read_bands(bands)

            print('Saving...')
            e.save_image(output_raster_uri, new_data, metadata=new_meta, interleave=out_interleave,
                         dtype=dtype)
            print('Done')

        return ENVIRaster(output_raster_uri)

    def edge_excision(self, output_raster_uri, data_ignore_value=0.0, dtype=None, out_interleave=None):
        """
        By using a bool mask, this method excise redundant edge of each band
        which exceed the intersection of spatial coverage of all bands.
        After this procedure, all bands (with the same resolution) in a single ASTER image will cover identical spatial extent.

        :param output_raster_uri:
        :param data_ignore_value:
        :param dtype:
        :param out_interleave:
        :return:
        """

        print('Excising ASTER redundant edge...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            nl, ns, nb = raster.shape

            meta['data ignore value'] = data_ignore_value

            if dtype is None:
                dtype = raster.dtype

            if out_interleave is None:
                out_interleave = meta['interleave']

            conditions = []
            for band in tqdm(range(nb), ncols=100, colour='white'):
                band_data = self.read_band(band)
                condition = np.logical_or(band_data == data_ignore_value, np.isnan(band_data))
                conditions.append(condition)

            conditions = np.any(conditions, axis=0)

            print('Excising...')
            new_data = np.zeros((nl, ns, nb), dtype=np.float32)
            for band in tqdm(range(nb), ncols=100, colour='white'):
                band_data = self.read_band(band)
                band_data[conditions] = data_ignore_value
                new_data[:, :, band] = band_data

            print('Saving...')
            e.save_image(output_raster_uri, new_data, metadata=meta, dtype=dtype, interleave=out_interleave)
            print('Done')

        return ENVIRaster(output_raster_uri)

    def radio_cali_edge_exc(self, output_raster_uri, data_ignore_value=0.0, out_interleave='bil',
                            vnsw_scale=0.1,
                            tir_scale=1,
                            dtype=np.float32):

        """
        :param output_raster_uri:
        :param data_ignore_value:
        :param vnsw_scale:
        :param tir_scale:
        :param dtype:
        :return:
        """

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            interleave = meta['interleave']
            nl, ns, nb = raster.shape
            gain = meta['data gain values']
            offset = meta['data offset values']

            print('Acquiring excision range...')

            conditions = []
            for band in tqdm(range(nb), ncols=100, colour='white'):
                band_data = self.read_band(band)
                condition = np.logical_or(band_data == data_ignore_value, np.isnan(band_data))
                conditions.append(condition)

            conditions = np.any(conditions, axis=0)

            print('Applying radiometric calibration and edge excision...')

            new_data = np.zeros((nl, ns, nb), dtype=np.float32)
            new_meta = meta.copy()
            for i in tqdm(range(nb), ncols=100, colour='white'):
                band_data = self.read_band(i)
                if i <= 8:
                    temp = vnsw_scale * (np.float32(gain[i]) * np.float32(band_data) + np.float32(offset[i]))
                else:
                    temp = tir_scale * (np.float32(gain[i]) * np.float32(band_data) + np.float32(offset[i]))
                temp[conditions] = data_ignore_value
                new_data[:, :, i] = temp

            new_meta['data ignore value'] = data_ignore_value
            new_meta['sensor type'] = 'ASTER'
            new_meta.pop('data gain values')
            new_meta.pop('data offset values')

            print(f'Saving as {out_interleave}...')
            e.save_image(output_raster_uri, new_data, metadata=new_meta, dtype=dtype, interleave=out_interleave)
            print('Done')

        return ENVIRaster(output_raster_uri)

    def thermal_atm_cor(self, output_raster_uri):

        print('Applying TIR atmospheric correction...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            task = envi.task('ThermalAtmosphericCorrection')
            input_raster = dict(url=self.fn.replace('.hdr', '.img'), factory='URLRaster')
            parameters = dict(
                INPUT_RASTER=input_raster,
                DATA_SCALE=1.0,
                FITTING_TECHNIQUE='Normalized Regression',
                OUTPUT_RASTER_URI=output_raster_uri.replace('.hdr', '.img')
            )
            task.execute(parameters)

            print('Done')

        return ENVIRaster(output_raster_uri)

    def modify_value(self, output_raster_uri, scale=1.0, offset=0.0, add=1e-9, max_val=None, min_val=None,
                     pattern='truncate', data_ignore_value=0.0, dtype=None):
        """
        result = truncated or normalized (scale * original data + offset)
        parameter add: is an infinitesimal compared with original data,
        when original non-background values are mapped to data_ignore_value, values = data_ignore_value + add,
        to avoid being ignored
        warning: when 'add' < 1, could be invalid with integer output

        parameter pattern: 'truncate': remove abnormal values exceed (min_val, max_val),
                            'normalize': linear stretch data values into (min_val, max_val)

        """

        print('Modifying values...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            interleave = meta['interleave']
            nl, ns, nb = raster.shape

            if dtype is None:
                dtype = raster.dtype

            data = self.load() * scale + offset

            """
            separate valid and background ranges
            """
            background_locations = (data == data_ignore_value)

            if pattern == 'truncate':
                if max_val is not None:
                    data[data > max_val] = max_val
                if min_val is not None:
                    data[data < min_val] = min_val
            elif pattern == 'normalize':
                """
                section need to be assigned when normalizing
                """
                if max_val is None:
                    raise VariableError('max_val')
                if min_val is None:
                    raise VariableError('min_val')
                for band in range(nb):
                    band_data = np.reshape(data[:, :, band], (nl, ns))
                    band_min = np.nanmin(band_data)
                    band_max = np.nanmax(band_data)
                    band_data = ((band_data - band_min) / (band_max - band_min)) * (max_val - min_val) + min_val
                    data[:, :, band] = band_data
            else:
                raise VariableError('pattern')

            data[data == data_ignore_value] = data_ignore_value + add
            data[background_locations] = data_ignore_value

            meta['data ignore value'] = data_ignore_value

            print('Saving...')
            e.save_image(output_raster_uri, data, metadata=meta, dtype=dtype, interleave=interleave)
            print('Done')

        return ENVIRaster(output_raster_uri)

    @staticmethod
    def layer_stack(input_rasters, output_raster_uri, sensor_type=None):
        """
        :param input_rasters: a list of input_raster objects
        :param output_raster_uri:
        :param sensor_type:
        :return:
        """
        print('Layer stacking...')

        input_raster_uris = [input_raster.fn for input_raster in input_rasters]

        input_raster_uris = [input_raster_uri.replace(get_fn_info(input_raster_uri).ext, 'img') for input_raster_uri in
                             input_raster_uris]
        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'img')

        if os.path.exists(output_raster_uri):
            print(output_raster_uri, 'already exists')
        else:
            task = envi.task('BuildLayerStack')
            input_rasters = [dict(url=input_raster_uri, factory='URLRaster') for input_raster_uri in input_raster_uris]
            parameters = dict(
                INPUT_RASTERS=input_rasters,
                RESAMPLING='Nearest Neighbor',
                OUTPUT_RASTER_URI=output_raster_uri
            )
            task.execute(parameters)

            if sensor_type:
                output_meta_fn = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')
                output_meta = e.open(output_meta_fn).metadata
                output_meta['sensor type'] = sensor_type
                e.write_envi_header(output_meta_fn, output_meta)

            print('Done')

        return ENVIRaster(output_raster_uri)

    def remove_edge(self, output_raster_uri, data_ignore_value=0.0, width=5):
        """
        Different with method 'edge_excision',
        this method is to apply a further clipping onto a preprocessed ASTER image
        whose bands are already aligned to the same extent (what 'edge_excision' did).

        Such subset is to deal with problems occur on edges/boundaries between valid values and background,
        like abnormal values or colors, by removing edges, surrounding the image, of a certain width.

        :param output_raster_uri:
        :param data_ignore_value:
        :param width:
        :return:
        """
        print('Removing outer edges...')

        if os.path.exists(output_raster_uri):
            print(output_raster_uri, 'already exists')
        else:
            task = envi.task('RemoveRasterBlackEdge')
            raster = self.raster
            meta = raster.metadata
            input_raster = dict(url=self.fn.replace('.hdr', '.img'), factory='URLRaster')
            parameters = dict(
                INPUT_RASTER=input_raster,
                DATA_IGNORE_VALUE=data_ignore_value,
                NUMBER_OF_PIXELS=width,
                OUTPUT_RASTER_URI=output_raster_uri.replace('.hdr', '.img')
            )
            task.execute(parameters)

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        print('Done')

        return ENVIRaster(output_raster_uri)

    def forward_pca(self, output_raster_uri, data_ignore_value=0.0, remove_background=False, keep_meta=False,
                    output_bands=None):
        """
        forward PCA transformation via spectral.principal_components() method
        Warning: will occupy a lot of memory (can be up to 32GB) when transforming GF-5/5B because of huge arrays

        :param data_ignore_value:
        :param remove_background:
        :param output_raster_uri:
        :param keep_meta: set True or False to keep or not keep metadata of input raster
        :param output_bands: an index list of output bands
        :return:
        """

        print('Applying PCA transformation...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            sta = None
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            data = self.load()

            """to acquire background range (for single band)"""
            band1 = self.read_band(0)
            background_locations = (band1 == data_ignore_value)
            band1 = None

            meta = raster.metadata
            nl, ns, nb = raster.shape
            interleave = meta['interleave']
            # 主成分分析变换，将主成分矩阵重新构造为多波段影像
            print('Transforming...')
            sta = spectral.principal_components(data)
            pca_data = sta.transform(data)
            # pca_data2, sta2 = pca_transform(data)

            data = None

            if output_bands is None:
                output_bands = range(nb)
            else:
                print('Taking selected bands...')
                pca_data = np.take(pca_data, output_bands, axis=2)

            if remove_background:
                for band in output_bands:
                    band_data = pca_data[:, :, band].reshape((nl, ns))
                    band_data[background_locations] = data_ignore_value
                    pca_data[:, :, band] = band_data

            new_meta = meta.copy()
            if not keep_meta:
                nl, ns, nb = raster.shape
                new_band_names = [f'PC {i + 1}' for i in range(nb)]
                new_meta['band names'] = new_band_names
                new_meta['description'] = 'PCA result'
                new_meta['data ignore value'] = data_ignore_value
                # Delete spectra-related keys
                spec_keys = ['wavelength', 'fwhm', 'wavelength units', 'data gain values', 'data offset values', 'bbl']
                for spec_key in spec_keys:
                    if spec_key in meta:
                        new_meta.pop(spec_key)

            for key in new_meta.keys():
                value = new_meta[key]
                if isinstance(value, list) and len(value) == nb:
                    new_meta[key] = [value[i] for i in output_bands]
                else:
                    new_meta[key] = value

            print('Saving...')
            """
            Saving will be bloody inefficient when perform PCA using method spectral.principal_components
            but faster when using self-customized pca transform function
            don't know why, both of the PCA results are the same numpy array
            spectral.principal_components can produce almost the same result as ENVI output
            """
            e.save_image(output_raster_uri, pca_data, metadata=new_meta, dtype=np.float32, interleave=interleave)
            print('Done')

        return ENVIRaster(output_raster_uri), sta

    def forward_pca_envi(self, output_raster_uri, data_ignore_value=0.0, remove_background=False, output_bands=None):

        """
        Compared with spectral.principal_components method, ENVI presents PCA API with much lower memory occupation,
        however ENVI doesn't offer 'output bands' parameters, thus we have to
        firstly let ENVI save all PC bands to local, then we reduce bands or remove background to produce a new raster,
        and replace the original ENVI output.

        It's an inefficient procedure because of repeated file saving and deleting operations, only recommended when
        you don't need to reduce bands or remove background, or you need sufficient memory to do other works.

        Statistics of forward PCA transformation won't be output as a single file but written to metadata using PCA API,
        and ENVI doesn't provide inverse PCA API, complete functions are available only in ENVI software.

        :param output_bands:
        :param remove_background:
        :param data_ignore_value:
        :param output_raster_uri:
        :return:
        """
        print('Applying PCA transformation...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            meta = self.raster.metadata

            """to acquire background range (for single band)"""
            band1 = self.read_band(0)
            background_locations = (band1 == data_ignore_value)
            band1 = None

            task = envi.task('ForwardPCATransform')
            input_raster = dict(url=self.fn.replace('.hdr', '.img'), factory='URLRaster')
            parameters = dict(
                INPUT_RASTER=input_raster,
                OUTPUT_RASTER_URI=output_raster_uri.replace('.hdr', '.img')
            )

            task.execute(parameters)

            if output_bands is not None:
                pca_raster = ENVIRaster(output_raster_uri)
                reduced_raster = pca_raster.export_bands(output_raster_uri.replace('_PCA', '_PCA2'), bands=output_bands)
                new_fn = reduced_raster.fn

                pca_raster = None
                reduced_raster = None

                os.remove(output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'img'))
                os.remove(output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr'))

                os.rename(new_fn.replace(get_fn_info(new_fn).ext, 'img'),
                          new_fn.replace(get_fn_info(new_fn).ext, 'img').replace('_PCA2.', '_PCA.'))
                os.rename(new_fn.replace(get_fn_info(new_fn).ext, 'hdr'),
                          new_fn.replace(get_fn_info(new_fn).ext, 'hdr').replace('_PCA2.', '_PCA.'))

            if remove_background:
                pca_raster = ENVIRaster(output_raster_uri)
                raster = pca_raster.raster
                nl, ns, nb = raster.shape
                meta = raster.metadata
                interleave = meta['interleave']
                new_data = np.zeros((nl, ns, nb), dtype=raster.dtype)
                for band in range(nb):
                    band_data = pca_raster.read_band(band)
                    band_data[background_locations] = data_ignore_value
                    new_data[:, :, band] = band_data

                e.save_image(output_raster_uri.replace('_PCA', '_PCA2'), new_data, metadata=meta,
                             interleave=interleave)

                new_fn = output_raster_uri.replace('_PCA', '_PCA2')

                raster = None
                pca_raster = None
                new_raster = None

                os.remove(output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'img'))
                os.remove(output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr'))

                os.rename(new_fn.replace(get_fn_info(new_fn).ext, 'img'),
                          new_fn.replace(get_fn_info(new_fn).ext, 'img').replace('_PCA2.', '_PCA.'))
                os.rename(new_fn.replace(get_fn_info(new_fn).ext, 'hdr'),
                          new_fn.replace(get_fn_info(new_fn).ext, 'hdr').replace('_PCA2.', '_PCA.'))

        return ENVIRaster(output_raster_uri)

    def inverse_pca(self, output_raster_uri, original_meta, sta):
        """
        based on what forward_pca returns, only available when all PC bands are exported
        :param output_raster_uri:
        :param original_meta: original metadata before forward PCA
        :param sta
        :return:
        """

        print('Applying inverse-PCA transformation...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            if sta is None:
                print('Statistics missed. Forward PCA needs to be redone to calculate statistics of transformation')
            else:
                pca_data = self.load()
                interleave = original_meta['interleave']

                # 逆变换，将主成分数据乘以主成分矩阵的转置，并加上均值
                inverse_transformed_data = np.dot(pca_data, sta.eigenvectors.T) + sta.mean
                # inverse_transformed_data = inverse_pca_transform(pca_data, sta)

                pca_data = None

                print('Saving...')
                e.save_image(output_raster_uri, inverse_transformed_data, metadata=original_meta,
                             dtype=np.float32,
                             interleave=interleave)
                print('Done')

        return ENVIRaster(output_raster_uri)

    def forward_mnf_envi(self, output_raster_uri, output_sta_uri):
        """
        :param output_sta_uri:
        :param output_raster_uri:
        :return:
        """
        print('Applying MNF transformation...')

        output_sta_uri = output_sta_uri.replace(get_fn_info(output_sta_uri).ext, 'sta')
        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'img')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            nl, ns, nb = raster.shape
            task = envi.task('ForwardMNFTransform')
            input_raster = dict(url=self.fn.replace('.hdr', '.img'), factory='URLRaster')
            parameters = dict(
                INPUT_RASTER=input_raster,
                # DIFF_SUBRECT =
                OUT_NBANDS=nb,
                # INPUT_NOISE_FILE =
                # OUTPUT_NOISE_FILE =
                OUTPUT_STATS_FILE=output_sta_uri,
                OUTPUT_RASTER_URI=output_raster_uri
            )
            task.execute(parameters)

            print('Done')

        return ENVIRaster(output_raster_uri)

    def inverse_mnf_envi(self, output_raster_uri, sta_uri, original_meta=None):
        """
        Warning: ENVI cannot output a correct result using inverse MNF API, but no problems with ENVI software.
        There may be something wrong with this API.

        :param original_meta:
        :param sta_uri:
        :param output_raster_uri:
        :return:
        """
        print('Applying inverse MNF transformation...')

        sta_uri = sta_uri.replace(get_fn_info(sta_uri).ext, 'sta')
        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'img')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            nl, ns, nb = raster.shape
            task = envi.task('InverseMNFTransform')
            input_raster = dict(url=self.fn.replace('.hdr', '.img'), factory='URLRaster')
            # input_sta = dict(url=sta_uri, factory='URLStatistics')
            parameters = dict(
                INPUT_RASTER=input_raster,
                INPUT_STATS_FILE=sta_uri,
                OUT_NCOMPONENTS=nb,
                OUTPUT_RASTER_URI=output_raster_uri
            )
            task.execute(parameters)

            if original_meta:
                output_meta_fn = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')
                output_meta = original_meta
                e.write_envi_header(output_meta_fn, output_meta)

            print('Done')

        return ENVIRaster(output_raster_uri)

    def moment_matching(self, output_raster_uri, width=None):
        """
        :param output_raster_uri:
        :param width:
        :return:
        """

        print('Moment matching...')

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            interleave = meta['interleave']
            nl, ns, nb = raster.shape

            if width is None:
                width = int(ns)

            # data = cp.asarray(raster.load()).astype(cp.float32)
            matched_data = np.zeros((nl, ns, nb), dtype=np.float32)

            for band in tqdm(range(nb), ncols=100, colour='white'):
                band_data = np.float32(self.read_band(band))
                matched_data[:, :, band] = match(band_data, width)

            print('Saving...')
            e.save_image(output_raster_uri, matched_data, metadata=meta, dtype=np.float32,
                         interleave=interleave)
            print('Done')

        return ENVIRaster(output_raster_uri)

    def denoise(self, output_raster_uri, data_ignore_value=0.0, width=None, dtype=None, out_interleave=None):
        """
        PCA -> moment matching -> inverse PCA
        :param data_ignore_value:
        :param dtype:
        :param out_interleave:
        :param width:
        :param output_raster_uri:
        :return:
        """

        output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            raster = self.raster
            meta = raster.metadata
            nl, ns, nb = raster.shape

            data = self.load()
            background_locations = (data == data_ignore_value)

            if width is None:
                width = int(ns)
            if dtype is None:
                dtype = raster.dtype
            if out_interleave is None:
                out_interleave = meta['interleave']

            # 主成分分析变换，将主成分矩阵重新构造为多波段影像
            print('Forward PCA transforming...')
            sta = spectral.principal_components(data)
            pca_data = sta.transform(data)

            data = None

            print('Moment matching...')
            matched_data = np.zeros((nl, ns, nb), dtype=np.float32)

            for band in tqdm(range(nb), ncols=100, colour='white'):
                band_data = np.reshape(pca_data[:, :, band], (nl, ns))
                matched_data[:, :, band] = match(band_data, width)

            print('Inverse PCA transforming...')
            # inverse_transformed_data = np.zeros((nl, ns, nb), dtype=np.float32)
            inverse_pca_data = np.ascontiguousarray(np.dot(matched_data, sta.eigenvectors.T) + sta.mean, dtype=dtype)
            inverse_pca_data[background_locations] = data_ignore_value
            # new_data = np.zeros((nl, ns, nb), dtype=np.float32)
            # for band in range(nb):
            #     new_data[:, :, band] = np.reshape(inverse_pca_data[:, :, band].astype(dtype), (nl, ns, nb))

            matched_data, sta = None, None

            print('Saving...')
            e.save_image(output_raster_uri, inverse_pca_data, metadata=meta,
                         dtype=dtype,
                         interleave=out_interleave)
            print('Done')

        return ENVIRaster(output_raster_uri)

    # def denoise_gpu(self, output_raster_uri, width=None, dtype=None, out_interleave=None):
    #     """
    #     (numba GPU accelerated)
    #     PCA -> moment matching -> inverse PCA
    #     :param dtype:
    #     :param out_interleave:
    #     :param width:
    #     :param output_raster_uri:
    #     :return:
    #     """
    #
    #     output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'hdr')
    #
    #     if os.path.exists(output_raster_uri):
    #         print(f'{output_raster_uri} already exists')
    #     else:
    #         raster = self.raster
    #         data = raster.load()
    #         meta = raster.metadata
    #
    #         if dtype is None:
    #             dtype = raster.dtype
    #         if out_interleave is None:
    #             out_interleave = meta['interleave']
    #
    #         denoise_data = denoise_gpu(data, width)
    #
    #         print('Saving...')
    #         e.save_image(output_raster_uri, denoise_data, metadata=meta,
    #                      dtype=dtype,
    #                      interleave=out_interleave)
    #         print('Done')

    def to_geotiff(self, output_raster_uri):
        """
        :param output_raster_uri:
        :return:
        """

        print('Converting ENVIRaster to GeoTIFF...')

        if os.path.exists(output_raster_uri):
            print(f'{output_raster_uri} already exists')
        else:
            # Open the ENVI raster
            src_ds = gdal.Open(self.fn.replace('.hdr', '.img'))
            output_raster_uri = output_raster_uri.replace(get_fn_info(output_raster_uri).ext, 'tif')

            # Set up output GeoTIFF driver
            driver = gdal.GetDriverByName("GTiff")

            # Convert to GeoTIFF
            gdal.Translate(output_raster_uri, src_ds, format="GTiff")

            # Close the datasets
            src_ds = None

            print('Done')

    def band_ratio(self, output_raster_uri, bands=(-1, -1), dtype=None, data_ignore_value=0.0):
        pass
