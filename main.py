import pandas as pd
import tabula
from action import integrate
import os
import shutil

# lattice=Trueでテーブルの軸線でセルを判定
os.mkdir("./files")
os.mkdir("./output_files")
dfs = tabula.read_pdf("000876235.pdf", lattice=True, pages='all', pandas_options={'header': None})
i = 0
for df in dfs:
    i += 1
    df = df.replace('\n', '', regex=True).replace('\r', '', regex=True).replace('\r\n', '', regex=True).replace('\n\r', '', regex=True)
    print(df)
    df.to_csv(f"./files/{i}.csv", index=None)

integrate("./files", "output_files")

shutil.rmtree("./files")
