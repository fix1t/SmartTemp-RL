import re
import os
import argparse

# Function to parse configuration data
def parse_hp_configurations(data, isDql=False):
    if isDql:
        pattern = r"dql_lr_(?P<lr>\d+\.\d+)_bs_(?P<bs>\d+)_df_(?P<df>\d+\.\d+)_nu_(?P<nu>\d+)_in_(?P<it>\d+\.\d+)"
    else:
        pattern = r"ppo_lr_(?P<lr>\d+\.\d+)_bs_(?P<bs>\d+)_df_(?P<df>\d+\.\d+)_nu_(?P<nu>\d+)_cl_(?P<cl>\d+\.\d+)"

    scores_pattern = r"Average score of last 10: (?P<score>-?\d+\.\d+)"
    entries = []
    for line in data.split("\n"):
        print(f"Line: {line}")
        config_match = re.search(pattern, line)
        score_match = re.search(scores_pattern, line)
        print
        if config_match and score_match:
            lr = config_match.group('lr')
            bs = config_match.group('bs')
            df = config_match.group('df')
            nu = config_match.group('nu')
            score = score_match.group('score')
            if isDql:
                it = config_match.group('it')
                entries.append((lr, bs, df, nu, it, score))
            else:
                cl = config_match.group('cl')
                entries.append((lr, bs, df, nu, cl, score))
    return entries

# Function to parse neural network configuration data
def parse_nn_configurations(data):
    pattern = r"Configuration: nn_(\d+(?:_\d+)*)\.yaml, Average score of last 10: (-?\d+\.\d+)"
    entries = []
    for line in data.split("\n"):
        match = re.search(pattern, line)
        if match:
            config = match.group(1).replace('_', ':')
            score = match.group(2)
            entries.append((config, score))
    return entries

def parse_configurations(data, nn=False, isDql=False):
    if nn:
        return parse_nn_configurations(data)
    return parse_hp_configurations(data, isDql)

# Function to generate LaTeX table with wrapped columns
def generate_latex_table(file, nn, isDql, rows_per_column=30, header=None):
    data = get_summary_data_from_file(file)
    entries = parse_configurations(data, nn, isDql)
    print(f"Entries: {entries}")

    if nn:
        header = "Network Layers & Score"
        # column_format = "|c|c"
    else:
        if isDql:
            header = "Learning Rate & Batch Size & Discount Factor & N Updates & Iterations & Score"
        else:
            header = "Learning Rate & Batch Size & Discount Factor & N Updates & Clip & Score"

    number_of_columns = header.count('&') + 1
    column_format = "|c" * number_of_columns

    num_columns = (len(entries) + rows_per_column - 1) // rows_per_column
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{" + "|".join([column_format] * num_columns) + "|}\n"
    latex += "\\hline\n"
    latex += " & ".join([header] * num_columns) + " \\\\\n"
    latex += "\\hline\n"
    for i in range(rows_per_column):
        row_data = []
        for j in range(num_columns):
            index = i + j * rows_per_column
            if index < len(entries):
                print(f"Index: {index}")
                if nn:
                    config, score = entries[index]
                    row_data.append(f"{config} & {score}")
                else:
                    row = ""
                    for item in entries[index]:
                        # item & item & item
                        row += item + " & " if item != entries[index][-1] else item
                    row_data.append(row)
            else:
                num_empty = 2 if nn else 4  # Number of columns in each configuration
                row_data.append(" & ".join([""] * num_empty))  # Empty cell for alignment
        latex += " & ".join(row_data) + " \\\\\n"
        latex += "\\hline\n"

    latex += "\\end{tabular}\n"
    latex += "\\caption{Configurations and their average scores}\n"
    latex += "\\label{tab:config_scores}\n"
    latex += "\\end{table}\n"
    return latex

def get_summary_data_from_file(file_path):
    try:
        with open(file_path) as f:
            data = f.read()
    except FileNotFoundError:
        print(f"File {file_path} not found")
        exit(1)
    return data

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Generate LaTeX table from configuration data')
    arg_parser.add_argument('file', type=str, help='File with configuration data')
    arg_parser.add_argument('--output_folder', type=str, default='out/tables', help='Output file for LaTeX table')
    arg_parser.add_argument('--rpc', type=int, default=50, help='Number of rows per column in LaTeX table')
    arg_parser.add_argument('-nn', action='store_true', help='Flag to parse neural network configurations')
    args = arg_parser.parse_args()

    data = get_summary_data_from_file(args.file)

    os.makedirs(args.output_folder, exist_ok=True)

    latex_table = generate_latex_table(data, args.nn, args.rpc)
    with open(f"{args.output_folder}/table.tex", "w") as f:
        f.write(latex_table)

    print(f"LaTeX table generated at {args.output_folder}/table.tex")
