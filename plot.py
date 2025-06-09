import json
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

def plot_radar(speedup, colors, ax=None):
    categories = [
        "Translation",
        "Multi-turn Conversation",
        "Retrieval-augmented Generation",
        "Mathematical Reasoning",
        "Question Answering",
        "Summarization",
    ]
    n_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]       # close the loop

    series = ["Lookahead", "REST", "PLD",
              "Token Recycling", "SAM", "Cacheback"]

    # Set up the radar chart
    ax.set_theta_offset(np.pi / 2)      # start at top
    ax.set_theta_direction(-1)          # clockwise

    # Customize position for each category label
    label_angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    label_positions = {
        "Translation": (label_angles[0], "Translation", 2.85),
        "Multi-turn Conversation": (label_angles[1], "Multi-turn\nConversation", 3.32),
        "Retrieval-augmented Generation": (label_angles[2], "Retrieval-\naugmented\nGeneration", 3.30),
        "Mathematical Reasoning": (label_angles[3], "Mathematical\nReasoning", 2.85),
        "Question Answering": (label_angles[4], "Question\nAnswering", 3.20),
        "Summarization": (label_angles[5], "Summarization", 3.35)
    }

    ax.set_xticks(label_angles)

    # Set each label position individually using different radial distances
    for angle, label, radius in label_positions.values():
        ax.text(angle, radius, label,
                ha='center', va='center',
                fontfamily='Times New Roman',
                fontsize=18,
                transform=ax.transData)
    ax.set_xticklabels([])  # Hide default labels

    ax.set_ylim(1, 2.4)
    ax.set_yticks([1, 1.5, 2, 2.4])
    ax.set_yticklabels(
        [1, 1.5, 2, 2.4], fontfamily='Times New Roman', fontsize=16)
    ax.set_rlabel_position(0)

    # Plot each series
    for idx, label in enumerate(series):
        vals = [speedup[label][category] for category in categories]
        vals = vals + vals[:1]          # close polygon
        ax.plot(
            angles,
            vals,
            linewidth=1.5,
            # marker=markers[idx % len(markers)],
            marker="o",
            markersize=4,
            label=label,
            color=colors[label]
        )
        ax.fill(angles, vals, alpha=0.07, color=colors[label])


def plot_bars(avg_speedup, colors, ax=None):
    # Data for the bar plot
    models = ['Vicuna 7B', 'Vicuna 13B', 'Vicuna 33B']

    n_methods = len(avg_speedup)
    x = np.arange(len(models)) * 0.7  # Reduced spacing between groups
    bar_width = 0.10  # Width of each individual bar

    # Plot each method's bars
    for idx, (method, values) in enumerate(avg_speedup.items()):
        offsets = x + (idx - n_methods / 2) * bar_width + bar_width / 2
        bars = ax.bar(offsets, values, width=bar_width,
                      label=method, color=colors[method])

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            if height == 0:
                text_height = 1
                text = 'X'
            else:
                text_height = height
                text = f"{height:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_height+0.015,
                text,
                ha='center',
                va='bottom',
                fontsize=16,
                fontfamily="Times New Roman"
            )

    # Customize axes and title
    ax.set_ylabel('Average Speedup Ratio', fontsize=18,
                  fontfamily="Times New Roman")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=18, fontfamily="Times New Roman")
    ax.set_ylim(1.0, 2.0)
    ax.tick_params(axis='y', labelsize=18, labelfontfamily="Times New Roman")

    ax.legend(
        bbox_to_anchor=(0.5, 1.15),
        loc='center',
        frameon=False,
        ncol=3,
        prop={'family': 'Times New Roman', 'size': 14}
    )


def plot_combined_benchmark():
    """Create a combined figure with both radar and bar plots"""

    colors = {
        "PLD": "#4E79A7",
        "SAM": "#F28E2B",
        "Lookahead": "#8C8C8C",
        "Token Recycling": "#B07AA1",
        "REST": "#59A14F",
        "Cacheback": "#E41A1C"
    }

    methods = ["Lookahead", "REST", "PLD",
               "Token Recycling", "SAM", "Cacheback"]

    speedups = {}
    for method in methods:
        speedups[method] = get_wall_clock_speedup(
            method, "7", "plot_data/model_answer")

    avg_speedup = {}
    for method in methods:
        avg_7b = get_wall_clock_speedup(
            method, "7", "plot_data/model_answer")["Average"]

        # Lookahead is not supported for 13B and 33B
        if method != "Lookahead":
            avg_13b = get_wall_clock_speedup(
                method, "13", "plot_data/model_answer")["Average"]
            avg_33b = get_wall_clock_speedup(
                method, "33", "plot_data/model_answer")["Average"]
        else:
            avg_13b = 0
            avg_33b = 0
        avg_speedup[method] = [avg_7b, avg_13b, avg_33b]

    # Create a figure with two subplots with different widths
    fig = plt.figure(figsize=(15, 4))

    # Create a gridspec to control subplot sizes
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 5])

    # Add subplot for radar plot (left side)
    ax1 = fig.add_subplot(gs[0], polar=True)
    plot_radar(speedups, colors, ax1)

    # Add subplot for bar plot (right side)
    ax2 = fig.add_subplot(gs[1])
    plot_bars(avg_speedup, colors, ax2)

    # Remove individual plot legend
    ax2.get_legend().remove()

    # Add a single legend spanning the entire figure
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.05),
        frameon=False,
        ncol=6,
        prop={'family': 'Times New Roman', 'size': 18}
    )

    # Adjust layout and save combined figure
    plt.tight_layout()
    plt.savefig("combined_plots.pdf", bbox_inches='tight')
    plt.show()


def get_wall_clock_time(path):
    model_answers = []
    with open(path, "r") as f:
        for line in f:
            model_answers.append(json.loads(line))

    classification = {
        "coding": "Multi-turn Conversation",
        "extraction": "Multi-turn Conversation",
        "humanities": "Multi-turn Conversation",
        "math": "Multi-turn Conversation",
        "reasoning": "Multi-turn Conversation",
        "roleplay": "Multi-turn Conversation",
        "stem": "Multi-turn Conversation",
        "writing": "Multi-turn Conversation",
        "math_reasoning": "Mathematical Reasoning",
        "rag": "Retrieval-augmented Generation",
        "qa": "Question Answering",
        "translation": "Translation",
        "summarization": "Summarization",
    }

    wall_clock_time = {
        "Multi-turn Conversation": {},
        "Retrieval-augmented Generation": {},
        "Mathematical Reasoning": {},
        "Question Answering": {},
        "Summarization": {},
        "Translation": {},
    }

    for answer in model_answers:
        category = classification[answer["category"]]
        ques_id = answer["question_id"]
        wall_clock_time[category][ques_id] = sum(
            answer["choices"][0]["wall_time"])

    return wall_clock_time


def get_wall_clock_speedup_from_file(path, baseline_path, exclude_ques_id=None):
    wall_clock_time = get_wall_clock_time(path)
    baseline_wall_clock_time = get_wall_clock_time(baseline_path)

    speedups = {}
    for category in wall_clock_time.keys():
        total_time = 0
        base_total_time = 0

        for ques_id in wall_clock_time[category].keys():
            if exclude_ques_id is not None and ques_id in exclude_ques_id:
                continue

            total_time += wall_clock_time[category][ques_id]
            base_total_time += baseline_wall_clock_time[category][ques_id]

        speedup = base_total_time / total_time
        speedups[category] = speedup

    speedups["Average"] = sum(speedups.values()) / len(speedups)
    return speedups


def get_wall_clock_speedup(method, model_size, dir_path):
    if method == "PLD":
        file_name = f"vicuna-{model_size}b-v1.3-pld-float16.jsonl"
    elif method == "SAM":
        file_name = f"vicuna-{model_size}b-v1.3-samd-float16.jsonl"
    elif method == "Cacheback":
        file_name = f"vicuna-{model_size}b-v1.3-cacheback-float16.jsonl"
    elif method == "Token Recycling":
        file_name = f"vicuna-{model_size}b-v1.3-recycling-float16.jsonl"
    elif method == "REST":
        file_name = f"vicuna-{model_size}b-v1.3-rest-float16-temperature-0.0-top_p-0.jsonl"
    elif method == "Lookahead":
        file_name = f"vicuna-{model_size}b-v1.3-lade-level-5-win-7-guess-7-float16.jsonl"
    else:
        raise ValueError(f"Invalid method: {method}")

    baseline_file_name = f"vicuna-{model_size}b-v1.3-baseline-float16.jsonl"

    path = os.path.join(dir_path, file_name)
    baseline_path = os.path.join(dir_path, baseline_file_name)
    return get_wall_clock_speedup_from_file(path, baseline_path)


def plot_heatmap():
    dir_path = "plot_data/shotgun_96_cr16_answer"

    grid = []
    for leader in range(1, 6):
        row = []
        for follower in range(1, 6):
            file_name = f"vicuna-7b-v1.3-shotgun-float16-p{leader}-f{follower}-cr16.jsonl"
            baseline_file_name = f"vicuna-7b-v1.3-baseline-float16.jsonl"
            path = os.path.join(dir_path, file_name)
            baseline_path = os.path.join(dir_path, baseline_file_name)
            speedup = get_wall_clock_speedup_from_file(path, baseline_path)
            row.append(speedup["Average"])
        grid.append(row)

    data = grid[::-1]  # Reverse the data rows to match the reversed y-axis
    num_row = len(data)
    num_col = len(data[0])

    plt.rcParams['pdf.fonttype'] = 42

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='bwr', vmin=1.0, vmax=2.0)

    # Add colorbar with manual ticks
    cbar = plt.colorbar(im)
    cbar.set_ticks([1.0, 1.25, 1.5, 1.75, 2.0])
    cbar.ax.invert_yaxis()  # Only invert the colorbar axis
    cbar.ax.set_ylabel('Speedup Ratio', fontfamily='Times New Roman', size=24)
    cbar.ax.tick_params(labelsize=20)
    for label in cbar.ax.get_yticklabels():
        label.set_family('Times New Roman')

    # Annotate each cell with its numeric value
    for i in range(num_row):
        for j in range(num_col):
            ax.text(j, i, f"{data[i][j]:.2f}", ha='center', va='center', fontfamily='Times New Roman', size=20)

    # Set tick marks to show grid indices
    ax.set_xticks(np.arange(num_col))
    ax.set_yticks(np.arange(num_row))
    ax.set_xticklabels(np.arange(1, num_col+1), fontfamily='Times New Roman', size=20)
    ax.set_yticklabels(np.arange(5, 0, -1), fontfamily='Times New Roman', size=20)
    ax.set_xlabel('Follower Length (FL)', fontfamily='Times New Roman', size=24)
    ax.set_ylabel('Leader Length (LL)', fontfamily='Times New Roman', size=24)

    plt.tight_layout()
    plt.savefig("heatmap.pdf", bbox_inches='tight')
    plt.show()

plot_combined_benchmark()
plot_heatmap()