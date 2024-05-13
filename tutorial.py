# ENVIRasterProcessV2 批处理模版
# 仅支持ENVI标准格式（*.hdr + *.img/dat）输入，输出也均为ENVI标准格式（*.hdr + *.img）
# FLAASH、热红外大气校正需要ENVI版本不低于5.7

# 导入必要的依赖
import os
import time
import numpy as np
from pprint import pprint
from imageprocess import *  # 工程自带的包
from utils import *  # 工程自带的包

"""主程序入口"""
if __name__ == '__main__':

    """
    首先定义批量影像的输入和输出路径input_dir、work_dir，后者保存的是每一步的输出结果，请确保留有足够空间
    这里的示例中input_dir和work_dir是拼接出来的，你可以直接定义，不必用root_dir
    如果不使用后续的steps和create_opath函数批量生成输出路径，而是自己定义每一步的输出路径，则不会用到work_dir
    """
    root_dir = r'H:\20240214_ASTER_Group1'
    input_dir = os.path.join(root_dir, 'ASTER_original_img')
    work_dir = os.path.join(root_dir, 'ASTER_processing')

    # 若work_dir在本地不存在，可令其自动创建
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    """
    然后使用get_fn_list函数检索input_dir下所有指定扩展名的文件，并返回一个带有绝对路径的文件名列表
    基于spectral.io.envi读写ENVI文件，扩展名需为hdr而不是img
    这种模式下有一个recursive参数，默认为False
    recursive=False表示仅检索input_dir根目录下的文件，
    recursive=True则会检索整个目录树，包括input_dir下的子文件夹
    """
    fn_list = get_fn_list(input_dir=input_dir, ext='hdr', recursive=False)

    """
    若不希望处理input_dir下的所有文件，提供一个csv格式的影像ID列表（不含扩展名），csv需要有表头（如filename）
    在这种模式下，get_fn_list将基于csv生成文件名列表
    显然，csv提供的文件名应该是input_dir下所有文件名的子集，若csv中包含input_dir下不存在的文件，之后也可被排查出来
    注：get_fn_list生成的每个文件名为f'{os.path.join(input_dir, basename)}.{ext}'
    若待检索的路径下的文件名非原始影像ID，而是带有后缀，如_pro，则csv列表中每个影像ID也需要添加_pro，才能拼接为正确的文件名
    """
    # csv_fn = r"H:\20240411_Group2_ASTER\filename_20240421.csv"
    # fn_list = get_fn_list(input_dir, ext='hdr', csv_fn=csv_fn, csv_header='filename')

    """
    初始化两个列表，false_list用于记录存在问题的影像文件名（这里的判断条件是波段数不足14，可以在下文中根据实际需求自己设置质检标准）
    no_exist_list用于记录csv中提及的、但在input_dir下不存在的影像文件名，若未使用csv检索，则该列表没有用处
    """
    false_list = []
    no_exist_list = []

    r"""
    定义步骤列表steps
    steps列表是配合下文中create_opath函数使用的，create_opath根据steps提供的信息创建一系列路径和生成一系列文件名
    如果前文中设置了work_dir，且希望使用steps+create_opath来批量创建输出名称，则steps的作用为：
    1.决定了每个步骤的输出存放的文件夹名称
    2.决定了每个步骤输出的影像文件的后缀
    3.直观展示做了哪些步骤
    """
    steps = \
        ['rad', ['VNSW_BIL', 'TIR_BIL'], ['fla', 'fla_mod', 'cloud', 'water', 'cor', 'cor_mod'], 'stack', 'exc', 'pro']
    r"""
    以这个steps为例，
    
    create_opath会在work_dir下创建一系列文件夹，名为：
    1_rad, 2_VNSW_BIL, 2_TIR_BIL, 3_fla, 3_fla_mod, 3_cloud, 3_water, 3_cor, 3_cor_mod, 4_stack, 5_exc, 6_pro
    
    默认为每个文件夹按步骤的先后顺序进行编号，被中括号括起来的步骤被认为是同一步，编号相同
    若不希望对文件夹进行编号，可在下文中将create_opath的参数serial_folder设为False
    
    create_opath还会根据当前处理的影像文件名，生成该影像每个步骤对应的输出文件名，并拼接为完整路径，返回一个字典（一系列键值对），
    例如当前处理的影像文件名为xxxxx.hdr，则返回的字典内容为：
    {
        'rad': '你设置的work_dir\1_rad\xxxxx_rad.hdr', （步骤编号不会被添加到影像文件名中）
        'VNSW_BIL': '你设置的work_dir\2_VNSW_BIL\xxxxx_VNSW_BIL.hdr',
        'TIR_BIL': '你设置的work_dir\2_TIR_BIL\xxxxx_TIR_BIL.hdr',
        以此类推
    }
    
    因此，对于每景影像的每个步骤的输出，我们可以方便地生成对应的带有绝对路径的文件名，并通过键值对索引到该文件名进行保存
    你可以按照自己的实际需求，自定义steps的内容
    """

    """-------------------------------------进入批处理循环-----------------------------------"""
    """遍历fn_list列表"""
    for idx, fn in enumerate(fn_list, start=1):
        basename = get_fn_info(fn).base  # 当前影像的文件名，不含路径和扩展名
        print(f'Processing {basename} | {idx}/{len(fn_list)} ...')  # 打印当前进度
        input_raster_uri = fn  # 当前影像带有绝对路径和扩展名的文件名

        """用来检查当前文件是否存在，若不存在则添加到no_exist_list，并跳过当前文件"""
        if not os.path.exists(input_raster_uri):
            print(f'{input_raster_uri} does not exist')
            no_exist_list.append(basename)
            continue

        """
        初始化ENVIRaster对象，并获取其相关信息
        ENVIRaster对象的创建方式为：
        变量名 = ENVIRaster(影像文件名)
        """
        envi_raster = ENVIRaster(input_raster_uri)  # 初始化ENVIRaster对象
        original_meta = envi_raster.raster.metadata  # 原始元数据
        acq_time = original_meta['acquisition time']  # 获取时间
        interleave = envi_raster.raster.interleave  # 存储方式
        nl, ns, nb = envi_raster.raster.shape  # nl, ns, nb分别为行数、列数、波段数

        """质检，可自定义质检标准，若不符合标准则添加到false_list，并跳过当前文件，此处检查的是ASTER波段是否完整"""
        if nb < 14:
            print(f'Incomplete Data ({nb} bands)')
            false_list.append(basename)
            continue

        r"""
        上文中，我们根据此轮循环当前的文件名input_raster_uri初始化了一个ENVIRaster对象，名为envi_raster，你可以取其他名字，然后即可开始应用其处理方法
        这些方法包括：
        read_band               读取单个波段，返回二维numpy数组，形状为（行数，列数）
        read_bands              读取多个波段，返回三维numpy数组，BSQ顺序（行数，列数，波段数），无论影像本身是什么存储顺序
        load                    读取全部波段，返回三维numpy数组，BSQ顺序（行数，列数，波段数），无论影像本身是什么存储顺序
        radio_cali              辐射定标，返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        edge_excision           取所有波段空间范围的交集作为该影像的空间范围，切除每个波段的冗余边缘，适用于ASTER。返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        radio_cali_edge_exc     辐射定标和边缘切除，整合为一个函数，适用于ASTER。返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        export_bands            单独导出某些波段，返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        flaash_atm_cor          ENVI FLAASH大气校正，返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        thermal_atm_cor         ENVI 热红外大气校正，返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        modify_value            调整影像数值，给定区间进行数值截断或线性拉伸，返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        remove_bad_bands        移除FLAASH标记出的坏波段，适用于AHSI。返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        rpc_ortho               ENVI 无参考的RPC正射校正，适用于AHSI。返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        layer_stack             静态方法，ENVI 图层堆栈（可自动重采样），返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        remove_edge             ENVI 黑边去除工具，用于裁剪掉影像外缘的一定宽度，适用于预处理后的ASTER。返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        forward_pca             正向PCA变换，返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        forward_pca_envi        基于ENVI的正向PCA变换，返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        inverse_pca             逆向PCA变换，与forward_pca绑定使用，返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        forward_mnf_envi        基于ENVI的正向MNF变换，返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        inverse_mnf_envi        基于ENVI的逆向MNF变换，但该API存在问题，不能给出正确结果，请使用ENVI软件
        moment_matching         矩匹配去噪，适用于AHSI。返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        denoise                 正向PCA -> 矩匹配去噪 -> 逆向PCA，适用于AHSI。返回ENVIRaster对象，且输出ENVI标准栅格到指定路径
        to_geotiff              将ENVIRaster转为GeoTIFF，输出至指定路径，无返回值
        
        上述方法除了layer_stack之外均为动态方法，调用方式为：
        变量名 = 某个ENVIRaster对象.method(*args, **kwargs)
        例如：
        envi_raster = ENVIRaster(input_raster_uri)
        rad = envi_raster.radio_cali(output_raster_uri, *args, **kwargs)
        由于大多数方法返回的仍然是ENVIRaster对象，我们可以连续运用动态方法：
        变量名 = ENVIRaster对象.method1().method2().method3()
        
        所有动态方法除了self之外的首个参数均为output_raster_uri，即输出的文件名，若我们已经用steps+create_opath创建了路径字典（假设名为opath），
        可以直接将output_raster_uri设为opath['xxx']，注意'xxx'步骤应已在steps内提供，大小写一致，否则字典中检索不到该键
        若你未创建路径字典，则需要自己设置输出路径output_raster_uri，例如：
        rad = envi_raster.radio_cali(r'xxx\xxx\xxx.hdr', *args, **kwargs)
        
        layer_stack是ENVIRaster类下的静态方法，调用方式为：
        变量名 = ENVIRaster.layer_stack(*args, **kwargs)
        第一个参数为包含多个ENVIRaster对象的列表：[object1, object2, ...]
        第二个参数为output_raster_uri
        """

        """开始计时"""
        start_time = time.time()

        """---------------------------开始处理--------------------------"""

        """
        创建路径字典opath
        若不希望对文件夹进行编号，可将serial_folder设为False
        """
        opath = create_opath(input_raster_uri, work_dir, steps, serial_folder=True)
        r"""
        使用create_opath创建路径字典后，我们可以直接通过键名索引到对应路径，
        例如下文中radio_cali_edge_exc要求输出路径时，我们可以直接将output_raster_uri设为opath['rad']，
        其值即为'你设置的work_dir\1_rad\xxxxx_rad.hdr'
        再次提醒，opath是基于steps的内容创建的，必须先在steps声明某个步骤名称'xxx'，opath中才会存在'xxx'这个键
        """

        """辐射定标、切边"""
        rad = envi_raster.radio_cali_edge_exc(opath['rad'], data_ignore_value=0.0, out_interleave='bil',
                                              vnsw_scale=0.1,
                                              tir_scale=1,
                                              dtype=np.float32)

        """导出VNSW（前9个波段），FLAASH大气校正，将数值截断至1~10000内"""
        vnsw = rad \
            .export_bands(opath['VNSW_BIL'], bands=range(9), out_interleave='bil') \
            .flaash_atm_cor(opath['fla'], opath['cloud'], opath['water'], acq_time, data_ignore_value=0.0,
                            atm_model='Tropical Atmosphere', sensor_type='ASTER') \
            .modify_value(opath['fla_mod'], max_val=10000, min_val=0, data_ignore_value=0.0)

        """导出TIR（9~14波段），热红外大气校正，将数值截断至0以上（没有设置最大值，只提供了min_val）"""
        tir = rad \
            .export_bands(opath['TIR_BIL'], bands=range(9, 14), out_interleave='bil') \
            .thermal_atm_cor(opath['cor']) \
            .modify_value(opath['cor_mod'], min_val=0, data_ignore_value=0.0)

        """
        将VNSW和TIR堆栈为一个影像，再次切边（其实波段已经对齐，这里是为了去掉TIR大气校正引入的非0值背景），
        然后进一步修剪掉影像四周30个像元的宽度
        """
        stack = ENVIRaster.layer_stack([vnsw, tir], opath['stack'], sensor_type='ASTER') \
            .edge_excision(opath['exc']) \
            .remove_edge(opath['pro'], data_ignore_value=0, width=30)

        """进一步编辑处理后影像的头文件信息"""
        output_fn = stack.fn  # stack是刚才处理完的ENVIRaster对象，属性fn为文件路径
        output_meta_fn = output_fn.replace(get_fn_info(output_fn).ext, 'hdr')  # 确保扩展名为hdr
        output_meta = e.open(output_meta_fn).metadata  # 打开该头文件
        """根据实际需求编辑头文件信息"""
        output_meta['acquisition time'] = original_meta['acquisition time']
        output_meta['data ignore value'] = 0.0
        output_meta['sensor type'] = 'ASTER'
        output_meta['unit for thermal infrared bands'] = 'W/m2/μm/sr'
        output_meta['reflectance scale factor'] = 10000.000000
        """将信息写入头文件"""
        e.write_envi_header(output_meta_fn, output_meta)

        """计时结束，打印一轮循环的耗费时间"""
        end_time = time.time()
        consume = format_time(end_time - start_time)
        print(f"Workflow consumed time: {consume}")

        """-------------------------------至此一轮循环结束-------------------------------"""

    """所有循环结束后，打印存在问题的数据或本地不存在的数据"""
    if false_list:
        pprint(false_list)
    if no_exist_list:
        pprint(no_exist_list)
