import time
from pathlib import Path
import re
from tqdm import tqdm

def get_subdir_list(p_sub_list):
    """
    To get a sub directory path list, Use thie func().
    
    pram: p_aub_list: specify a directory which sub dirs is gotten from.
    """
    # 引数の直下のディレクトリ・パスの一覧を取得
    sub_dir_list_temp = []
    for p_sub in tqdm(p_sub_list):
        sub_dir_list_temp.append([p_s_s for p_s_s in p_sub.iterdir()])
    # ２次元リストを１次元リスト化
    return sum(sub_dir_list_temp, [])


p = Path('../cif/')
p_sub_list = [p_s for p_s in p.glob('[0-9]')]


before = time.time()
cif_path_list = get_subdir_list(get_subdir_list(get_subdir_list(p_sub_list)))
after = time.time()
print(f"it took {after - before}sec to get all of subdir list.")


def cif_filter(cif_file_path):
    pattern = '.*\.cif'
    cif_file_path_str = str(cif_file_path)
    return bool(re.match(pattern, cif_file_path_str))


cif_path_list = [path for path in cif_path_list if cif_filter(path)]

