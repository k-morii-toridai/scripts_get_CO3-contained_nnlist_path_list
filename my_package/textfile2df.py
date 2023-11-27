import pandas as pd
# 表示オプションを設定
pd.set_option('display.float_format', '{:.6f}'.format)


def poscar2df_coords(filename="./POSCAR"):
    """
    Converting a POSCAR file to a dataframe.
    
    pram1: filename: POSCAR file path
    """
    # POSCARファイルの読み込み
    def poscar2df_xyz(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # 構造情報が始まる行を特定
        for i, line in enumerate(lines):
            if 'Direct' in line or 'Cartesian' in line:
                start_line = i + 1
                break

        # 原子座標データを取得
        atom_data = lines[start_line:]
        df_xyz = pd.DataFrame([line.split() for line in atom_data], columns=['x', 'y', 'z'])

        return df_xyz


    def poscar2df_species(filename):    
        # POSCARファイルを読み込む
        with open(filename, 'r') as file:
            lines = file.readlines()

        # 原子種とその数を含む行を抽出
        element_species = lines[5].split()
        element_count = [int(s) for s in lines[6].split()]

        # # 原子種のリストと対応する数のリストを抽出
        species_list = [elem for elem, count in zip(element_species, element_count) for _ in range(count)]

        # # 原子種とその数からPandasのSeriesを作成
        df_species = pd.DataFrame(species_list, columns=['Species'])

        return df_species


    df_species_added = pd.merge(poscar2df_xyz(filename), poscar2df_species(filename), left_index=True, right_index=True)
    df_central_atom_series = pd.DataFrame([i for i in range(1, len(df_species_added) + 1)], columns=['central atom'])
    df_central_atom_added = pd.merge(df_central_atom_series, df_species_added, left_index=True, right_index=True)
    
    
    return df_central_atom_added
    

    

def nnlist2df(POSCAR_nnlist='POSCAR.nnlist'): 
    
    # CSVファイルに書き込むためのファイルを作成
    output_csv_fname = POSCAR_nnlist + '.csv'
    
    def nnlist2csv(POSCAR_nnlist='POSCAR.nnlist'):
        # テキストファイルの内容を読み込みます
        with open(POSCAR_nnlist, 'r') as input_file:
            lines = input_file.readlines()

        # # CSVファイルに書き込むためのファイルを開きます
        # output_csv_fname = POSCAR_nnlist + '.csv'
        with open(output_csv_fname, 'w') as output_file:
            # CSVヘッダを書き込みます
            output_file.write("central atom,neighboring atom,distance,X,Y,Z,unitcell_x,unitcell_y,unitcell_z,central species,neighboring species\n")

            # テキストファイルの各行を処理します
            for line in lines:
                # スペースで区切られた各要素を取得します
                elements = line.split()

                # CSV行を構築します
                csv_line = ','.join(elements[:11]) + '\n'

                # CSVファイルに書き込みます
                output_file.write(csv_line)

    # nnlistを変換したcsvファイルを作成
    nnlist2csv(POSCAR_nnlist)
    df_nnlist = pd.read_csv(output_csv_fname)    
    
    return df_nnlist



if __name__ == "__main__":    
    filename="./POSCAR"
    print(poscar2df_coords(filename))