import os
import re
import sys
import itertools
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import numpy as np

from my_package.textfile2df import nnlist2df


# どの距離のPOSCAR.nnlistを，CO3を含むPOSCARかどうかの判定に使うのかを受け取る
args = sys.argv


def flatten_func(list_2dim):
    return list(itertools.chain.from_iterable(list_2dim))


def bool_CO3_contained_poscar(poscar_nnlist):
    df_nnlist = nnlist2df(str(poscar_nnlist))

    # df_nnlistでcentral speciesがCのものに絞る
    df_nnlist_central_species_C = df_nnlist[df_nnlist['central species'] == 'C']

    # さらに，炭酸イオンかどうかを判定するためにに，あるcentral atomのneighboring speciesがCとO３つの計４つでできているか確認したい
    # central atomの値を入力すれば，neighboring speciesのリストを返す関数()を作成
    def get_neighboring_species_list(central_atom_id, df=df_nnlist_central_species_C):
        """
        To get all central atoms of a cluster(:neighbors), Input a number of cluster center element number(:central atom)

        Input: central atom column element In df_nnlist
     -> Output: All neighboring atom column element that Input(:elemnt) match central atom column element

        param1: Input: central atom column element In df_nnlist
        """
        # 左側の列から対応する行を選択し、右側の数値を取得
        # result = df_nnlist[df_nnlist['central atom'] == input_value]['neighboring atom'].values
        neighboring_species_list = df[df['central atom'] == central_atom_id]['neighboring species'].tolist()
        return neighboring_species_list

    def match_num_C_O_3(central_atom_id):
        # 原子団の要素数は，Cが1つ，Oが3つの計4つかどうかcheck
        if len(get_neighboring_species_list(central_atom_id)) == 4:
            # 原子団の要素にCが1つだけ含まれているかどうかcheck
            if get_neighboring_species_list(central_atom_id).count('C') == 1:
                # 原子団の要素にOがちょうど3つ含まれているかどうかcheck
                if get_neighboring_species_list(central_atom_id).count('O') == 3:
                    return True
            else:
                return False
        else:
            return False

    # df_nnlist_central_species_Cに対し，CO3がどうかを確認し，CO3である原子のid一覧を取得
    # まず，中心元素がCのid一覧(central atomの値の一覧)を取得
    central_species_C_id_list = df_nnlist_central_species_C['central atom'].unique()
    # その中で，match_num_C_O_3()を用いて，過不足なくCO3だけを含むものに絞る
    num_matched_central_species_C_id_list = [i for i in central_species_C_id_list if match_num_C_O_3(i)]

    def rm_duplicated_central_species_C_id(central_species_C_list=num_matched_central_species_C_id_list):

        def get_neighboring_atom_id_list(central_atom_id, df=df_nnlist_central_species_C):
            """
            To get all central atoms of a cluster(:neighbors), Input a number of cluster center element number(:central atom)

            Input: central atom column element In df_nnlist
         -> Output: All neighboring atom column element that Input(:elemnt) match central atom column element

            param1: Input: central atom column element In df_nnlist
            """
            # 左側の列から対応する行を選択し、右側の数値を取得
            # result = df_nnlist[df_nnlist['central atom'] == input_value]['neighboring atom'].values
            neighboring_atom_id_list = df[df['central atom'] == central_atom_id]['neighboring atom'].tolist()

            return neighboring_atom_id_list

        # 中心がC，その周りにOが3つ存在する原子のid一覧をリスト化
        species_id_list = flatten_func(list(map(get_neighboring_atom_id_list, num_matched_central_species_C_id_list)))
        # 直上のspecies_id_listの重複削除してリスト化
        set_species_id_list = list(set(species_id_list))
        # species_id_listの要素から，set_species_id_listの要素を削除 -> species_id_listにはCO3ではない原子のリストが入る
        for num in set_species_id_list:
            species_id_list.remove(num)

        # species_id_listの要素数が0 -> Oの重複抽出なし -> すべてCO3の炭酸イオン（：C2O6でない）
        if len(species_id_list) == 0:
            return True
        else:
            # central_atom_idをkey，そのneighboring atom idをvalueにして，CO3が塊まった形でデータ保持する
            central_species_C_dict = {}
            for i in central_species_C_list:
                central_species_C_dict[i] = get_neighboring_atom_id_list(i)
            # species_id_listにはCO3ではない原子のリストが入っている
            # CO3でない原子のidを含むCO3の塊はFalse, それ以外のCO3はTureを返し，リスト化
            CO3_filter_list = []
            for duplicated_num in species_id_list:
                CO3_filter_list.append([not (duplicated_num in value_list) for value_list in central_species_C_dict.values()])
            # 2重リストになっているCO3_filter_listを要素ごとに論理積（ex). [[True, False, True], [False, True, Ture]] -> [False, False, True]）
            CO3_filter = [min(t) for t in zip(*CO3_filter_list)]
            # match_C_O_3で絞ったcentral_species_C_listから，さらに重複してCO3になっているものを除去（ex). C2O6）
            CO3_matched_central_species_C_list = np.array(list(central_species_C_dict.keys()))[CO3_filter]

            return True if len(CO3_matched_central_species_C_list) > 0 else False

    return rm_duplicated_central_species_C_id()


def iterdir_func(poscar_dir):
    return list(poscar_dir.iterdir())


def folder_nnlist_filter(path):
    nnlist_dist_num = 'nnlist_' + args[1]  # arg[1]にはnnlist作成時に指定した距離を受け取る
    pattern = f'{nnlist_dist_num}$'  # 正規表現（：末尾が'nnlist_{arg[1]}'で終わる）
    string = str(path)
    return bool(re.search(pattern, string))


def poscar_nnlist_filter(path):
    pattern = 'POSCAR.nnlist$'  # 正規表現（：末尾が'POSCAR.nnlist'で終わる）
    string = str(path)
    return bool(re.search(pattern, string))


# C_O_existed_pos_nnlist_path_listを.npyからload
npy_path = 'scripts_get_C_O_existed_poscar_abs_path_list/C_O_existed_poscar_folder_path_list.npy'
C_O_existed_poscar_folder_path_list_loaded = np.load(npy_path, allow_pickle=True)


p = Pool(cpu_count() - 1)
try:
    # iterdir
    C_O_existed_poscar_nnlist_path_list = list(tqdm(p.imap(iterdir_func, C_O_existed_poscar_folder_path_list_loaded), total=len(C_O_existed_poscar_folder_path_list_loaded)))
    # flatten
    C_O_existed_poscar_nnlist_path_list = flatten_func(C_O_existed_poscar_nnlist_path_list)
    # make filter
    folder_nnlist_filter = list(tqdm(p.imap(folder_nnlist_filter, C_O_existed_poscar_nnlist_path_list), total=len(C_O_existed_poscar_nnlist_path_list)))
    # cast list to ndarray
    C_O_existed_poscar_nnlist_path_list = np.array(C_O_existed_poscar_nnlist_path_list)
    # apply filter to ndarray
    C_O_existed_poscar_nnlist_path_list = C_O_existed_poscar_nnlist_path_list[folder_nnlist_filter]
finally:
    p.close()
    p.join()


p = Pool(cpu_count() - 1)
try:
    # iterdir
    C_O_existed_poscar_nnlist_path_list = list(tqdm(p.imap(iterdir_func, C_O_existed_poscar_nnlist_path_list), total=len(C_O_existed_poscar_nnlist_path_list)))
    # flatten
    C_O_existed_poscar_nnlist_path_list = flatten_func(C_O_existed_poscar_nnlist_path_list)
    # make filter
    poscar_nnlist_filter = list(tqdm(p.imap(poscar_nnlist_filter, C_O_existed_poscar_nnlist_path_list), total=len(C_O_existed_poscar_nnlist_path_list)))
    # cast list to ndarray
    C_O_existed_poscar_nnlist_path_list = np.array(C_O_existed_poscar_nnlist_path_list)
    # apply filter to ndarray
    C_O_existed_poscar_nnlist_path_list = C_O_existed_poscar_nnlist_path_list[poscar_nnlist_filter]
finally:
    p.close()
    p.join()


print("Now, judging whether C_O_existed_poscar file is CO3-contained or not.")
p = Pool(cpu_count() - 1)
try:
    # make filter
    CO3_contained_nnlist_filter = list(tqdm(p.imap(bool_CO3_contained_poscar, C_O_existed_poscar_nnlist_path_list), total=len(C_O_existed_poscar_nnlist_path_list)))
    # apply filter to ndarray
    CO3_contained_nnlist_path_list = np.array(C_O_existed_poscar_nnlist_path_list)[CO3_contained_nnlist_filter]
finally:
    p.close()
    p.join()

print(f"len(CO3_contained_nnlist_path_list)/len(C_O_existed_poscar_nnlist_path_list)):\
{len(CO3_contained_nnlist_path_list)}/{len(C_O_existed_poscar_nnlist_path_list)}")

# make CO3-contained poscar file and folder list from CO3-contained POSCAR.nnlist list
CO3_contained_poscar_folder_path_list = [Path(os.path.split(os.path.split(p)[0])[0]) for p in CO3_contained_nnlist_path_list]
CO3_contained_poscar_path_list = [Path(str(p) + '/POSCAR') for p in CO3_contained_poscar_folder_path_list if os.path.exists(Path(str(p) + '/POSCAR'))]

# CO3を含むPOSCARファイルの親ディレクトリパスのリストを.npy形式で保存
np.save(f'CO3_contained_poscar_folder_path_list_{args[1]}.npy', CO3_contained_poscar_folder_path_list)
# CO3を含むPOSCARファイルのパスを.npy形式で保存
np.save(f'CO3_contained_poscar_path_list_{args[1]}.npy', CO3_contained_poscar_path_list)
