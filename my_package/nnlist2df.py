import pandas as pd
# 表示オプションを設定
pd.set_option('display.float_format', '{:.6f}'.format)



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