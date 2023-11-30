import os 
import sys
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import numpy as np

from my_package.textfile2df import nnlist2df


def bool_CO3_contained_poscar(POSCAR_nnlist):
    df_nnlist = nnlist2df(POSCAR_nnlist)
    
    # df_nnlistでcentral speciesがCのものに絞る
    df_nnlist_central_species_C = df_nnlist[df_nnlist['central species'] == 'C']
    
    # さらに，炭酸イオンかどうかを判定するためにに，あるcentral atomのneighboring speciesがCとO３つの計４つでできているか確認したい
    # central atomの値を入力すれば，neighboring speciesのリストを返す関数get_neighboring_species_list()を作成
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
    

    def match_C_O_3(central_atom_id):
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
    central_species_C_list = df_nnlist_central_species_C['central atom'].unique()
    # その中で，match_C_O_3()を用いて，過不足なくCO3だけを含むものに絞る
    matched_central_species_C_list = [i for i in central_species_C_list if match_C_O_3(i)]
     
    return  True if len(matched_central_species_C_list) > 0 else False



args = sys.argv

# C_O_existed_pos_nnlist_path_listを.npyからload
C_O_existed_poscar_folder_abs_path_list_loaded = np.load(args[1], allow_pickle=True)

# POSCAR.nnlistの絶対パスのリストを作成
C_O_existed_pos_nnlist_abs_path_list = [str(folder) + '/POSCAR.nnlist' for folder in C_O_existed_poscar_folder_abs_path_list_loaded]

# 実際にそのパス（ファイル）が存在するするか確認
print(f"Now, check existence of C_O-existed POSCAR.nnlist...")
p = Pool(cpu_count() - 1)
try:
    bool_exist_or_not = list(tqdm(p.imap(os.path.exists, C_O_existed_pos_nnlist_abs_path_list), total=len(C_O_existed_pos_nnlist_abs_path_list)))
    
finally:
    p.close()
    p.join()


if set(bool_exist_or_not):
    print("C_O_existed_pos_nnlist_abs_path_list's path is all existence!")

else:
    print(f"Not exist: {set(bool_exist_or_not) - {True}}")

# C_O_existed_pos_nnlist_path_list_loaded = np.load('../scripts_mk_C_O_contained_poscar_filter/C_O_existed_pos_nnlist_path_list.npy', allow_pickle=True)


print(f"Now, judging whether C_O_existed_poscar file is CO3-contained or not.")
p = Pool(cpu_count() - 1)
try:
    bool_CO3_contained_nnlist_list = list(tqdm(p.imap(bool_CO3_contained_poscar, C_O_existed_pos_nnlist_abs_path_list), total=len(C_O_existed_pos_nnlist_abs_path_list)))

finally:
    p.close()
    p.join()

    
# CO3_contained_nnlist_abs_path_listを.npy形式で保存
CO3_contained_nnlist_abs_path_list = np.array(C_O_existed_pos_nnlist_abs_path_list)[bool_CO3_contained_nnlist_list]
print(f"len(CO3_contained_nnlist_abs_path_list/len(C_O_existed_pos_nnlist_abs_path_list)): {len(CO3_contained_nnlist_abs_path_list)}/{len(C_O_existed_pos_nnlist_abs_path_list)}")
np.save('CO3_contained_nnlist_abs_path_list.npy', CO3_contained_poscar_folder_abs_path_list)
