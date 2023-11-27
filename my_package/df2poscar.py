import os


# 元のPOSCARファイルから5行目までを抽出して、新しいPOSCARファイルに書き込む関数
def df2poscar(df, original_file="./POSCAR", output_file="./gen_data/POSCAR"):
    """
    Writing the DataFrame(:df_coords_fix) to a POSCAR file.
    param1: df=df_coords_fix: DataFrame that has 'X', 'Y', 'Z' columns about coords.
    param2: original POSCAR file
    param3: generated POSCAR file
    """
    
    # df_coords_fixを文字列に変換
    def df2str(df):
        df_coords_fix_str = df[['X', 'Y', 'Z']].to_string(header=False, index=False, index_names=False)
        return df_coords_fix_str

    
    # df_coords_fixから元素種を文字列として抽出する関数
    def return_species(df):
        species_line = ' '.join(df['Species'].unique())
        num_line = ' '.join([str(len(df[df['Species'] == specie])) for specie in df['Species'].unique()])
        return species_line + '\n' + num_line

    
    # 元のPOSCARファイルの5行目までを抽出し，新しいファイルに書き込む
    def wirte_header2poscar():
        # 最初の5行を抽出
        with open(original_file, 'r') as infile:
            lines = infile.readlines()[:5]
        # 新しいPOSCARファイルに書き込む
        with open(output_file, 'w') as outfile:
            outfile.writelines(lines)
    
    
    # 新しいPOSCARファイルに書き込んでいく
    def write_species2poscar():
        with open(output_file, 'a') as file:
            # すでに存在するテキストファイルに元素種を追記
            file.write(return_species(df) + '\n')
            # 元素種まで書かれたファイルにDirectという文字をを追記
            file.write('Direct\n')
            # 直交座標を追記
            file.write(df_coords_fix_str + '\n')


    # 関数をcall
    None if os.path.exists("gen_data") else os.makedirs('gen_data')
    df_coords_fix_str = df2str(df)
    wirte_header2poscar()
    write_species2poscar()
    
    print(f"{output_file} にクラスタ化後の内容が書き込まれました。")
