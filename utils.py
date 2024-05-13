import os
import fnmatch
import pandas as pd
import random


class VariableError(Exception):
    def __init__(self, variable_name):
        self.variable_name = variable_name

    def __str__(self):
        return f"Error in variable or parameter: '{self.variable_name}'. Could be None or unexpected value"


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)} h {int(m)} m {int(s)} s"


def serial(lst):
    """
   :param lst: a string list (can be nested) of each procedure ['xxx', 'xxx', ['xxx', 'xxx'], 'xxx', ...]
   :return: a numbered 1-D string list ['1_xxx', '2_xxx', '3_xxx', '3_xxx', '4_xxx', ...]
   """
    result = []
    for i, item in enumerate(lst, start=1):
        if isinstance(item, list):
            result.extend(f'{i}_{ele}' for ele in item)
        else:
            result.append(f'{i}_{item}')
    return result


def in_list(ele, lst):
    """
    :param ele:
    :param lst:
    :return:
    """
    if ele in lst:
        return True
    else:
        for item in lst:
            if isinstance(item, list):
                if in_list(ele, item):
                    return True
    return False


def get_fn_info(fn):
    """
    :param fn: a filename string with absolute path
    :return: an object with attributes fn, dir, base, ext
    """
    fn = str(fn)
    # 使用os.path模块获取目录、基本名称和扩展名
    dir = os.path.dirname(fn)
    basename = os.path.basename(fn)
    # basename, ext = os.path.splitext(basename)

    # Split the filename based on '.'
    parts = basename.split('.')

    # Extract directory, base name, and extension
    base = '.'.join(parts[:-1])  # Join all parts except the last one
    ext = parts[-1] if len(parts) > 1 else ''

    class FileInfo:
        def __init__(self, fn, dir, base, ext):
            self.fn = fn
            self.dir = dir
            self.base = base
            self.ext = ext

    return FileInfo(fn, dir, base, ext)


def create_opath(fn, out_dir, steps, ext=None, serial_folder=True, simplify_basename=False):
    """
    :param fn:
    :param out_dir: output directory
    :param fn: processed filename. to help generating output filename
    :param steps: procedure list. to help generating output folder of each step
    :param ext: extension of output filename. default 'hdr'
    :param serial_folder:
    :param simplify_basename:
    :return:
    """
    result_paths = {}

    if serial_folder:
        steps = serial(steps)

    for step in steps:
        step_dir = os.path.join(out_dir, step)

        # 如果目录不存在，则创建
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)

        # 生成输出文件路径
        if ext is None:
            ext = 'hdr'

        if serial_folder:
            postfix = '_'.join(step.split('_')[1:])
        else:
            postfix = '_'.join(step.split('_')[:])

        basename = get_fn_info(fn).base
        if simplify_basename:
            basename = '_'.join(basename.split('_')[0:5])

        filename = f"{basename}_{postfix}.{ext}"
        output_path = os.path.join(step_dir, filename)

        # 将结果添加到字典
        result_paths[postfix] = output_path

    return result_paths


def find_xml_files(input_dir):
    """
    This function has been eliminated
    """
    # 初始化一个列表来存储找到的文件名
    xml_files = []

    # 递归遍历目录树
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # 检查文件是否以 '.xml' 后缀结尾且文件名中不包含 'Check' 字符串
            if 'Check' not in filename and 'describe' not in filename and filename.lower().endswith('.xml'):
                # 添加符合条件的文件名到结果列表
                xml_files.append(file_path)

    # 返回找到的 XML 文件名列表
    return xml_files


def file_search(input_dir, ext=None, recursive=False):
    """
    :param input_dir: input directory for search
    :param ext: extension of target file type. default 'hdr'
    :param recursive: set True or False to decide if search recursively
    :return: a filename list with absolute path
    """
    if ext is None:
        ext = 'hdr'

    if recursive:
        fn_list = []
        for dirpath, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if filename.lower().endswith('.' + ext):
                    # 添加符合条件的文件名到结果列表
                    fn_list.append(file_path)
    else:
        # 使用 fnmatch 模块匹配文件
        fn_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                   os.path.isfile(os.path.join(input_dir, f))
                   # and (fnmatch.fnmatch(f, '*.dat') or fnmatch.fnmatch(f, '*.img'))]
                   and (fnmatch.fnmatch(f, f'*.{ext}'))]
    # xml_files = find_xml_files(input_dir)
    # fn_list = files + xml_files

    # fn_list = files

    if fn_list:
        return fn_list
    else:
        # print('No standard ENVIRaster[*.dat/*.img] or [*.xml] file of China Satellites found.')
        print(f'No {ext} file found.')
        return []


def get_fn_list(input_dir, ext=None, recursive=False, csv_fn=None, csv_header='filename'):
    """
    :param input_dir: 
    :param ext: 
    :param recursive: 
    :param csv_fn: 
    :param csv_header: 
    :return: 
    """
    if ext is None:
        ext = 'hdr'

    if csv_fn is None:
        return file_search(input_dir, ext, recursive)
    else:
        fn_list = []
        basename_list = pd.read_csv(csv_fn)[csv_header]
        for basename in basename_list:
            # basename = '_'.join(basename.split('_')[0:5])
            input_raster_uri = f'{os.path.join(input_dir, basename)}.{ext}'
            fn_list.append(input_raster_uri)
        return fn_list


def create_color_table(num=1):
    # 生成20种随机颜色
    color_table = []
    for _ in range(num):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        color_table.extend([red, green, blue])

    return color_table


if __name__ == '__main__':
    color_table = create_color_table(20)
    print(color_table)
