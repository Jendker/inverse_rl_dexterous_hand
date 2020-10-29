import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import numpy as np
import os
import click
import shutil


result_paths = [['job_name_1', 'visible name 1'],
                ['job_name_2', 'visible name 2'],
                ['job_name_3', 'visible name 3']]

category = ''
fields = ['success_rate']
print_function_field = ''
print_function = np.max
print_last_iteration_field = ''
x_label = 'Iterations'
y_labels = []  # will be inferred if empty
legend_title = ''
title = ''
custom_axes_scales = {}
x_limit = 0
field_to_normalize = ''
average_over_max_runs = None

skip_same_iterations = False
draw_segments_where_repeated = False
calculate_average_of_runs = True
smoothen = 10
force_std_dev = False

skip_places = {}
linestyles = ['-', '--']
colors = []  # example colors = ['#1f77b4', '#2ca02c']


def update_colors(line_count):
    from cycler import cycler
    colors = plt.cm.inferno(np.linspace(0, 0.8, line_count))
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)


def update_globals(variables):
    import importlib
    module = 'inverse_rl_dexterous_hand.utils.results_variables.' + variables.replace('/', '.')
    imported = importlib.import_module(module)
    globals().update(imported.__dict__)


def get_longest_substring(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and is_substring(data[0][i:i+j], data):
                    substr = data[0][i:i+j]
    first_string = data[0]
    index = first_string.find(substr)
    last_index = index + len(substr)
    if index != 0:
        while first_string[index-1] != '_' and index < len(first_string):
            index += 1
    if last_index != len(first_string):
        while first_string[last_index] != '_' and last_index > 0:
            last_index -= 1
    return first_string[index:last_index]


def get_run_names(runs_orig, empty_name=''):
    runs = runs_orig.copy()
    while len(get_longest_substring(runs)) > 3:
        longest_substr = get_longest_substring(runs)
        longest_substr.strip('_')
        runs = [run.replace(longest_substr, '') for run in runs]
        runs = [run.replace('__', '_') for run in runs]
    runs = [run.replace('_', ' ') for run in runs]
    for index, run in enumerate(runs):
        if not run:
            runs[index] = 'standard' if not empty_name else empty_name
    return runs


def is_substring(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True


def read_category(category_name, runs_folder, name_if_empty=''):
    global category, result_paths
    result_paths = []
    category = category_name
    category_path = '../training/' + runs_folder + 'CATEGORIES/' + category
    files_list = [f for f in sorted(os.listdir(category_path)) if os.path.isfile(os.path.join(category_path, f)) and '.py' in f]
    if len(files_list) == 1:
        category_file_script = files_list[0]
        assert '.py' in category_file_script
        import importlib
        module = 'inverse_rl_dexterous_hand.training.' + runs_folder[:-1] + '.CATEGORIES.' + category + '.' + \
                 category_file_script.replace('.py', '')
        imported = importlib.import_module(module.replace('/', '.'))
        globals().update(imported.__dict__)
    if not result_paths:
        runs_dirs = [f for f in sorted(os.listdir(category_path)) if os.path.isdir(os.path.join(category_path, f))
                     and f != '__pycache__']
        run_names = get_run_names(runs_dirs, name_if_empty)
        result_paths = [[run_dir, name] for run_dir, name in zip(runs_dirs, run_names)]
        globals().update({'result_paths': result_paths})
    if os.path.exists(os.path.join(category_path, '__pycache__')):
        shutil.rmtree(os.path.join(category_path, '__pycache__'))
    return category


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
    """
    orig_lenth = len(x)
    x = np.array(x)
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[:orig_lenth]


def parse_single_line(log_path, field):
    line = []
    with open(log_path, 'r') as log_file:
        reader = csv.DictReader(log_file, delimiter=',')
        last_iteration = -1
        if field not in reader.fieldnames:
            print("Warning: field", field, "does not exist in the log:", log_path)
            print("Will be fed with 0s.\n")
        for index, row in enumerate(reader):
            if field not in row:  # non existing fields in files are fed with 0s after warning
                this_row_value = 0
            else:
                this_row_value = float(row[field])
            iteration = int(row['iteration'])
            if iteration == last_iteration:
                if skip_same_iterations:
                    line[-1] = this_row_value
                else:
                    if row['iteration'] not in skip_places:
                        skip_places[row['iteration']] = len(line)
                    line.append(this_row_value)
            else:
                line.append(this_row_value)
            last_iteration = iteration
            if x_limit:
                if index == x_limit:
                    break
    return line


def get_lines_for_pair(pair, field, runs_folder):
    training_folder = pair[0]
    if category:
        training_folder = 'CATEGORIES/' + category + '/' + training_folder
    training_path = '../training/' + runs_folder + training_folder
    # check if there are multiple runs
    listdir = os.listdir(training_path)
    run_folders = [element for element in listdir if 'run_' in element]
    if not run_folders:
        run_folders = ['.']
    run_folders.sort()
    run_lines = []
    for run_folder in run_folders[:average_over_max_runs]:
        log_path = os.path.join(training_path, run_folder, 'logs/log.csv')
        run_line = parse_single_line(log_path, field)
        run_lines.append(run_line)
    assert all(len(run_line) == len(run_lines[0]) for run_line in run_lines), \
        'A run line length different for other lines. training_folder: ' + training_folder
    return run_lines, run_folders


def parse_lines(runs_folder):
    lines = {}
    for field in fields:
        lines[field] = []
    for field in fields:
        for pair in result_paths:
            run_lines, run_folders = get_lines_for_pair(pair, field, runs_folder)
            legend = pair[1]
            if calculate_average_of_runs:
                line_mean_values = np.mean(run_lines, axis=0)
                line_std_dev_values = np.std(run_lines, axis=0)
                lines[field].append({'mean': line_mean_values, 'std_dev': line_std_dev_values,
                                     'legend': legend})
            else:
                for run_folder, run_line in zip(run_folders, run_lines):
                    lines[field].append([run_line, legend + '_' + run_folder])
    if print_function_field:
        for pair in result_paths:
            run_lines, _ = get_lines_for_pair(pair, print_function_field, runs_folder)
            legend = pair[1]
            print(legend + ':', np.mean(print_function(run_lines, axis=1)))
    elif print_last_iteration_field:
        for pair in result_paths:
            run_lines, _ = get_lines_for_pair(pair, print_last_iteration_field, runs_folder)
            legend = pair[1]
            print(legend + ':', np.mean(np.array(run_lines)[:, -1]))
    return lines


# https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
def normalize_field(field_name, lines):
    for key in lines:
        if key == field_name:
            for i, _ in enumerate(lines[key]):
                lines[key][i]['mean'] = (lines[key][i]['mean'] - np.min(lines[key][i]['mean'])) / np.ptp(lines[key][i]['mean'])
                lines[key][i]['std_dev'] = (lines[key][i]['std_dev'] - np.min(lines[key][i]['std_dev'])) / np.ptp(lines[key][i]['std_dev'])
    return lines


@click.command()
@click.option('--variables', type=str, help='specify the script name with variables to read')
@click.option('--category_name', type=str, help='specify category with variables to load')
@click.option('--name_if_empty', '--empty', type=str, help='given name to an run if common string becomes empty')
@click.option('--draw_std_dev', '--std_dev', is_flag=True, help='draws standard deviations between runs')
def main(variables, category_name, name_if_empty, draw_std_dev):
    if x_limit > 0:
        print("Warning: x_limit active, set to:", x_limit)
    assert not(variables and category_name), 'Define either variables or category_name'
    if print_function_field:
        print("Values for field:", print_function_field, "with function used over line:", print_function.__name__)
    elif print_last_iteration_field:
        print("Mean values of the last iteration for the field:", print_last_iteration_field)
    runs_folder = 'Runs/'
    global category
    if category_name:
        category = read_category(category_name, runs_folder, name_if_empty)
    elif variables:
        update_globals(variables)

    lines = parse_lines(runs_folder)
    # update_colors(line_count=len(result_paths))

    if smoothen:
        for field, field_lines in lines.items():
            for index, dictionary in enumerate(field_lines):
                if len(dictionary['mean']) < smoothen:
                    print("Line length is shorter than window length for smoothing. Please check that. Exiting.")
                    exit(1)
                field_lines[index]['mean'] = smooth(dictionary['mean'], window_len=smoothen, window='flat')
                field_lines[index]['std_dev'] = smooth(dictionary['std_dev'], window_len=smoothen, window='flat')

    if field_to_normalize:
        print("Warning: normalizing field", field_to_normalize)
        lines = normalize_field(field_name=field_to_normalize, lines=lines)

    fig, ax_host = plt.subplots()
    axes = {}
    for index, key in enumerate(lines):
        if index == 0:
            axes[key] = ax_host
        else:
            axes[key] = ax_host.twinx()
    plot_lines = []
    colors_index = 0
    for index, field in enumerate(lines):
        for line in lines[field]:
            points = line['mean']
            legend = line['legend']
            x = list(range(len(points)))
            y = points
            plot_line_array = axes[field].plot(x, y, colors[colors_index] if colors else '', linestyle=linestyles[index], label=legend)
            colors_index += 1
            if draw_std_dev or force_std_dev:
                std_dev = line['std_dev']
                axes[field].fill_between(x, y + std_dev, y - std_dev, alpha=0.3)
            plot_lines.append(plot_line_array[0])

    if draw_segments_where_repeated:
        for value in skip_places.values():
            ax_host.axvline(x=value, ymin=0, ymax=0.25, color='red', lw='0.3')

    for index, key in enumerate(lines):
        if not y_labels:
            this_y_label = key.replace('_', ' ').capitalize()
        else:
            this_y_label = y_labels[index].capitalize()
        if this_y_label.lower() == 'success rate':
            this_y_label += ' [%]'
        if len(lines) > 1:
            this_y_label += ' (' + linestyles[index] + ')'
        if index == 0:
            axes[key].set(xlabel=x_label, ylabel=this_y_label)
        else:
            axes[key].set(ylabel=this_y_label)
    if legend_title:
        legend = ax_host.legend(title=legend_title)
    else:
        legend = ax_host.legend()
    ax_host.grid()
    if title:
        ax_host.set_title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    lined = dict()
    texts = dict()
    line_to_text = dict()
    for legline, legtext, origline in zip(legend.get_lines(), legend.get_texts(), plot_lines):
        legline.set_picker(5)  # 5 pts tolerance
        legtext.set_picker(5)
        lined[legline] = origline
        texts[legtext] = legline
        line_to_text[legline] = legtext

    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        event_element = event.artist
        if event_element in texts:
            legline = texts[event_element]
        else:
            legline = event_element
        legtext = line_to_text[legline]
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
            legtext.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
            legtext.set_alpha(0.2)
        fig.canvas.draw()

    for axis_name, scale_type in custom_axes_scales.items():
        axes[axis_name].set_yscale(scale_type)

    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    fig.savefig("fig.png")

if __name__ == '__main__':
    main()
