import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMEOUT = 3600
plt.style.use('ggplot')
fs = 15  # font size
ticks_fs = 15
lw = 2.5  # line width
figsize = (10, 8)
label_step = 50


def error(msg=""):
    print(msg)
    sys.exit(1)


def get_input_files(input_dirs):
    input_files = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            error("Input folder '{}' does not exists".format(input_dir))
        for filename in os.listdir(input_dir):
            filepath = os.path.join(input_dir, filename)
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(filepath)
                if ext == ".json":
                    input_files.append(filepath)
    if not input_files:
        error("No .json file found")
    input_files = sorted(input_files)
    print("Files found:\n\t{}".format("\n\t".join(input_files)))
    return input_files


def parse_inputs(input_files):
    data = []
    for filename in input_files:
        with open(filename) as f:
            result_out = json.load(f)
        mode = result_out["mode"]
        if mode == "PANL":
            continue

        params = result_out["params"]
        # print("File: {:80s} found: {:3d}".format(filename, len(result_out["results"])))

        for result in result_out["results"]:
            result["time"] = min(result["time"], TIMEOUT)
            if result["time"] == TIMEOUT and "integration_time" in result:
                result["integration_time"] = TIMEOUT
            # assert result["time"] >= result["integration_time"]
            result["mode"] = mode
            result.update(params)
        data.extend(result_out["results"])

    data = pd.DataFrame(data)

    modes = (data["mode"].unique())
    sort_by = [("time", mode) for mode in ["XSDD", "XADD", "PA", "PAEUF", "PAEUFTA"]
               if mode in modes]
    # groupby to easily generate MulitIndex
    data = data. \
        groupby(["support", "weight", "mode"]). \
        aggregate({
            "time": "min",
            "n_integrations": "min",
            "value": "min",
            "depth": "min",
            "real": "min",
            "bool": "min",
            # "integration_time": "sum"
        }). \
        unstack()
    data['time'] = data['time'].fillna(TIMEOUT)
    # do not plot n_integrations where mode times out
    data['n_integrations'] = data['n_integrations'].where(
        data['time'] < TIMEOUT, pd.NA)
    data.sort_values(by=sort_by, inplace=True)
    return data


def plot_data(outdir, data, param, timeout=None, frm=None, to=None, filename=""):
    total_problems = len(data)
    if frm is not None and to is not None:
        data = data[frm:to]
        sfx = "_{}_{}".format(frm, to)
    elif frm is not None:
        data = data[frm:]
        sfx = "_{}_{}".format(frm, total_problems)
    elif to is not None:
        data = data[:to]
        sfx = "_{}_{}".format(0, to)
    else:
        sfx = ""

    data[param].plot(linewidth=lw,
                     figsize=figsize)
    n_problems = len(data)
    # timeout line
    if timeout is not None:
        x = list(range(n_problems))
        y = [timeout] * n_problems
        plt.plot(x,
                 y,
                 linestyle="dashed",
                 linewidth=lw,
                 label="timeout",
                 color="r")

    if param == "time":
        ylabel = "Query execution time (seconds)"
    else:
        ylabel = "Number of integrations"

    # legend
    plt.legend(loc="center left", fontsize=fs)
    # axes labels
    plt.xlabel("Random problem instances", fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    # xticks
    positions = list(range(0, n_problems, label_step))
    labels = list(range(frm or 0, to or total_problems, label_step))
    plt.xticks(positions, labels, fontsize=ticks_fs)
    plt.yticks(fontsize=ticks_fs, rotation=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    outfile = os.path.join(
        outdir, "{}_uai{}{}.png".format(param, sfx, filename))
    plt.savefig(outfile)
    print("created {}".format(outfile))
    plt.clf()


def check_values(data):
    ii = data["time", "PA"] < TIMEOUT

    for mode in data.columns.get_level_values(1).unique():
        ii_m = data[ii]["time", mode] < TIMEOUT
        diff = ~np.isclose(data[ii][ii_m]["value", mode].values,
                           data[ii][ii_m]["value", "PAEUF"].values)
        if diff.any():
            print("Error! {}/{} values of {} do not match with PA".format(
                diff.sum(), len(diff), mode))
            print(data[ii][ii_m][diff].reset_index()[["time", "value"]])
        else:
            print("Mode {:10s}: {:4d} values ok".format(
                mode, len(data[ii][ii_m])))


def parse_interval(interval):
    frm, to = interval.split("-")
    frm = int(frm) if frm != "" else None
    to = int(to) if to != "" else None
    return frm, to


def main(args):
    inputs = args.input
    output_dir = args.output
    intervals = args.intervals
    filename = args.filename
    timeout = None if args.no_timeout else TIMEOUT

    if not os.path.exists(output_dir):
        error("Output folder '{}' does not exists".format(output_dir))

    input_files = get_input_files(inputs)
    data = parse_inputs(input_files)
    check_values(data)
    for interval in intervals:
        frm, to = parse_interval(interval)
        plot_data(output_dir, data, "time", frm=frm, to=to, filename=filename)
        plot_data(output_dir, data, "n_integrations",
                  frm=frm, to=to, filename=filename)

    plot_data(output_dir, data, "time", timeout=timeout, filename=filename)
    plot_data(output_dir, data, "n_integrations", filename=filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot WMI results')
    parser.add_argument(
        'input', nargs='+', help='Folder and/or files containing result files as .json')
    parser.add_argument('-o', '--output', default=os.getcwd(),
                        help='Output folder where to put the plots (default: cwd)')
    parser.add_argument('-f', '--filename', default="",
                        help='String to add to the name of the plots (optional)')
    parser.add_argument('--intervals', nargs='+', default=[],
                        help='Sub-intervals to plot in the format from-to (optional)')
    parser.add_argument('--no-timeout', action='store_true',
                        help='If true timeout line is not plotted')
    main(parser.parse_args())
