import pandas as pd
import tabula
from action import integrate
import os
import shutil

if not os.path.exists("./files"):
    os.mkdir("./files")

if not os.path.exists("./output_files"):
    os.mkdir("./output_files")

for i in range(1, 47):
    opendata_file = os.listdir(f"./data_files/shinryoujo_{i}")
    dfs = tabula.read_pdf(f"./data_files/shinryoujo_{i}/{opendata_file[0]}", lattice=True, pages='all', pandas_options={'header': None})
    j = 0
    for df in dfs:
        j += 1
        df = df.replace('\n', '', regex=True).replace('\r', '', regex=True).replace('\r\n', '', regex=True).replace('\n\r', '', regex=True)
        print(df)
        if j == 1:
            df_header = df.iloc[:2]
            print(df_header)
        df.to_csv(f"./files/{j}.csv", index=None)
    integrate("./files", f"./output_files/output{i}.csv")
    shutil.rmtree("./files")
    os.mkdir("./files")

shutil.rmtree("./files")
