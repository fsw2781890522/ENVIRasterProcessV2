from imageprocess import *
from utils import *
from bandmath import rgb_combine

if __name__ == '__main__':

    """Silica,Fe2+指数波段组合涉及到MNF的波段，需要先对影像批量MNF变换"""

    input_dir = r'H:\20240508_chen\ASTER_processing\6_pro'  # 预处理后的影像
    mnf_dir = r'H:\20240508_chen\ASTER_processing\7_mnf'  # MNF影像
    work_dir = r'H:\20240508_chen\ASTER_processing\8_combine'  # 输出

    fn_list = get_fn_list(input_dir, ext='hdr', recursive=False)

    """要做的指数名称"""
    features = [
        'AlOH_minerals,advanced_argillic_alteration',
        'Clay,amphibole,laterite',
        'Gossan,alteration,host_rock(1)',
        'Decorellation',
        'Silica,carbonate',
        'Discrimination_for_mapping',
        'Discrimination_in_sulphide_rich_areas',
        'Silica,Fe2+',
        'Enhanced_structual_features']

    for idx, fn in enumerate(fn_list, start=1):
        basename = get_fn_info(fn).base  # 当前影像的文件名，不含路径和扩展名
        print(f'Processing {basename} | {idx}/{len(fn_list)} ...')  # 打印当前进度
        input_raster_uri = fn  # 当前影像带有绝对路径和扩展名的文件名
        input_mnf_uri = fn.replace(input_dir, mnf_dir).replace('_pro.', '_mnf.')  # 根据实际情况生成MNF影像的路径

        for feature in features:
            rgb_combine(input_raster_uri,
                        input_mnf_uri,
                        work_dir,
                        feature,
                        scale=10000,  # 放大倍数
                        dtype=np.int16)  # 放大10000倍后可以整型输出减少空间占用
