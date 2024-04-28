import re
import os
import argparse

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
def generate_latex_table(file, nn=False, rows_per_column=30, header=None):
    data = get_summary_data_from_file(file)
    entries = parse_configurations(data)
    if not entries:
        print(f"No configurations found in {file}")
        exit(1)

    header = header if header else " & ".join(key.replace('_', ' ').title() for key in entries[0].keys())

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
