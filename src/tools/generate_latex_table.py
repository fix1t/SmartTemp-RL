import re
import os
import argparse

# # Function to parse configuration data
# def parse_hp_configurations(data, isDql=False):
#     if isDql:
#         pattern = r"(?P<x1>\)(?P<lr>\d+\.\d+)_bs_(?P<bs>\d+)_df_(?P<df>\d+\.\d+)_nu_(?P<nu>\d+)_it_(?P<it>\d+\.\d+)"
#     else:
#         pattern = r"ppo_lr_(?P<lr>\d+\.\d+)_bs_(?P<bs>\d+)_df_(?P<df>\d+\.\d+)_nu_(?P<nu>\d+)_cl_(?P<cl>\d+\.\d+)"
#     scores_pattern = r"Average score of last 10: (?P<score>-?\d+\.\d+)"
#     entries = []
#     for line in data.split("\n"):
#         config_match = re.search(pattern, line)
#         score_match = re.search(scores_pattern, line)
#         if config_match and score_match:
#             lr = config_match.group('lr')
#             bs = config_match.group('bs')
#             df = config_match.group('df')
#             nu = config_match.group('nu')
#             score = score_match.group('score')
#             if isDql:
#                 it = config_match.group('it')
#                 entries.append((lr, bs, df, nu, it, score))
#             else:
#                 cl = config_match.group('cl')
#                 entries.append((lr, bs, df, nu, cl, score))
#     return entries

# # Function to parse neural network configuration data
# def parse_nn_configurations(data):
#     pattern = r"Configuration: nn_(\d+(?:_\d+)*)\.yaml, Average score of last 10: (-?\d+\.\d+)"
#     entries = []
#     for line in data.split("\n"):
#         match = re.search(pattern, line)
#         if match:
#             config = match.group(1).replace('_', ':')
#             score = match.group(2)
#             entries.append((config, score))
#     return entries

# def parse_configurations(data, nn=False, isDql=False):
#     if nn:
#         return parse_nn_configurations(data)
#     return parse_hp_configurations(data, isDql)


def parse_configurations(data):
    # Pattern to exactly match lines starting with "1. Configuration:" and similar
    config_line_pattern = r"^\d+\.\sConfiguration:"
    param_pattern = r"(\w+):(-?[\d\.]+|\[[-\d\.,\s]+\])"
    entries = []

    # Split data into lines and process each line
    for line in data.split("\n"):
        if re.match(config_line_pattern, line):
            # Remove '.yaml' and other unnecessary parts for cleaner parsing
            line = re.sub(r'\.yaml,.*', '', line)  # Removes the '.yaml' and everything after it
            # Find all matches of parameter-value pairs
            matches = re.findall(param_pattern, line)
            if matches:
                entry = {key: parse_value(value) for key, value in matches}
                entries.append(entry)
    return entries

def parse_value(value):
    # Helper function to parse values into float or list of floats
    if value.startswith('[') and value.endswith(']'):
        return [float(x.strip()) for x in value[1:-1].split(',')]
    return float(value)

# Function to generate LaTeX table with wrapped columns
def generate_latex_table(file, nn, rows_per_column=30, header=None):
    data = get_summary_data_from_file(file)
    entries = parse_configurations(data)

    header = header if header else " & ".join(entries[0].keys())

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
                row_data.append(" & ".join(str(value) for value in entries[index].values()))
            else:
                num_empty = 2 if nn else 5  # Number of columns in each configuration
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

    os.makedirs(args.output_folder, exist_ok=True)

    latex_table = generate_latex_table(args.file, args.nn, rows_per_column=args.rpc)
    with open(f"{args.output_folder}/table.tex", "w") as f:
        f.write(latex_table)

    print(f"LaTeX table generated at {args.output_folder}/table.tex")
# Example data string with multiple lines and varying numbers of name-value pairs
# import re

# def parse_dynamic_configurations(data):
#     # Updated pattern to include list values in brackets
#     pattern = r"(\w+):([\d\.]+|\[[\d\.,\s]+\])"
#     entries = []
#     for line in data.replace('+', ' ').split("\n"):
#         matches = re.findall(pattern, line)
#         if matches:
#             entry = {key: parse_value(value) for key, value in matches}
#             entries.append(entry)
#     return entries

# def parse_value(value):
#     # Helper function to parse values into float or list of floats
#     if value.startswith('[') and value.endswith(']'):
#         return [float(x.strip()) for x in value[1:-1].split(',')]
#     return float(value)

# # Example data string with lines having both scalar and list values
# data = """
# lr:0.01+bs:32+df:0.99+nu:5+cl:[10.2, 123.5]
# lr:0.02+bs:64+df:0.95+nu:3+it:7.5
# lr:0.03+bs:48+df:0.98+nu:4+cl:1]
# """

# # Using the function to parse the provided example data
# parsed_data = parse_dynamic_configurations(data)

# # Printing the results
# for config in parsed_data:
#     print(config)




data2 ="""
Summary:
Total time: 48 hours, 6 minutes, 48.79 seconds
Total configurations run 141 of 162

Configurations results from best to worst:
1. Configuration: dql_lr_0.0015_bs_128_df_0.99_nu_20_it_0.025.yaml, Average score of last 10: 93.82
2. Configuration: dql_lr_0.0005_bs_128_df_0.95_nu_20_it_0.025.yaml, Average score of last 10: -104.28
3. Configuration: dql_lr_0.0005_bs_96_df_0.95_nu_10_it_0.001.yaml, Average score of last 10: -209.94
4. Configuration: dql_lr_0.0005_bs_96_df_0.90_nu_5_it_0.005.yaml, Average score of last 10: -233.19
5. Configuration: dql_lr_0.0005_bs_128_df_0.95_nu_10_it_0.005.yaml, Average score of last 10: -420.48
"""
