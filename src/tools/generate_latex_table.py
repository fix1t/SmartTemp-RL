import re
import os
import argparse

# Function to parse configuration data
def parse_hp_configurations(data):
    pattern = r"cfg_lr_(?P<lr>\d+\.\d+)_bs_(?P<bs>\d+)_df_(?P<df>\d+\.\d+)"
    scores_pattern = r"Average score of last 10: (?P<score>-?\d+\.\d+)"

    entries = []
    for line in data.split("\n"):
        config_match = re.search(pattern, line)
        score_match = re.search(scores_pattern, line)
        if config_match and score_match:
            lr = config_match.group('lr')
            bs = config_match.group('bs')
            df = config_match.group('df')
            score = score_match.group('score')
            entries.append((lr, bs, df, score))
    return entries

# Function to generate LaTeX table with wrapped columns
def generate_latex_table(entries, rows_per_column=30):
    num_columns = (len(entries) + rows_per_column - 1) // rows_per_column
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|" + "|".join(["|c|c|c|c"] * num_columns) + "|}\n"
    latex += "\\hline\n"
    header = "LR & BS & DF & Score "
    latex += " & ".join([header] * num_columns) + " \\\\\n"
    latex += "\\hline\n"

    for i in range(rows_per_column):
        row_data = []
        for j in range(num_columns):
            index = i + j * rows_per_column
            if index < len(entries):
                lr, bs, df, score = entries[index]
                row_data.append(f"{lr} & {bs} & {df} & {score}")
            else:
                row_data.append(" & ".join([""] * 4))  # Empty cell for alignment
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
    args = arg_parser.parse_args()

    data = get_summary_data_from_file(args.file)

    os.makedirs(args.output_folder, exist_ok=True)

    entries = parse_hp_configurations(data)
    latex_table = generate_latex_table(entries, args.rpc)

    with open(f"{args.output_folder}/table.tex", "w") as f:
        f.write(latex_table)

    print(f"LaTeX table generated at {args.output_folder}/table.tex")
