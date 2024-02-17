import pandas as pd
import glob


def integrate(input_file_path, output_file_name):
    csv_files = glob.glob(f"{input_file_path}/*.csv")
    data_list = []

    for csv_file in csv_files:
        data_list.append(pd.read_csv(csv_file))

    df = pd.concat(data_list, axis=0, sort=True)

    df.to_csv(f"{output_file_name}", index=False)
