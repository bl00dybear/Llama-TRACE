import os
import logging
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)

def calculate_bwt(results_matrix, num_tasks):
    if num_tasks < 2:
        return 0.0
    total = 0.0
    for i in range(num_tasks - 1):
        total += results_matrix[num_tasks - 1][i] - results_matrix[i][i]
    return total / (num_tasks - 1)

def calculate_fwt(baselines, results_matrix, num_tasks):
    if num_tasks < 2:
        return None
    fwt_sum = 0.0
    for j in range(1, num_tasks):
        fwt_sum += results_matrix[j - 1][j] - baselines[j]
    return fwt_sum / (num_tasks - 1)

def calculate_op(results_matrix, num_tasks):
    last_row = results_matrix[num_tasks - 1]
    return sum(last_row[:num_tasks]) / num_tasks

def log_heatmap_to_wandb(results_matrix, cfg, baselines=None, fwt_per_task=None, step_label=None):
    if wandb is None or wandb.run is None:
        return

    tasks = list(cfg.data.tasks)
    num_eval_tasks = len(tasks)

    y_labels = []
    matrix_rows = []

    if baselines is not None:
        y_labels.append("Zero-shot")
        matrix_rows.append(list(baselines))

    for i, row in enumerate(results_matrix):
        y_labels.append(f"After T{i+1}")
        matrix_rows.append(list(row))

    table_data = []
    for row_idx, y_label in enumerate(y_labels):
        for col_idx in range(num_eval_tasks):
            table_data.append([
                f"T{col_idx+1} {tasks[col_idx]}",
                y_label,
                float(matrix_rows[row_idx][col_idx]),
            ])

    table = wandb.Table(
        data=table_data,
        columns=["Eval Task", "Train Step", "Score"],
    )
    wandb.log({"accuracy_heatmap": table})

    if plt is not None and np is not None:
        fig = _build_heatmap_figure(matrix_rows, y_labels, tasks, fwt_per_task)
        wandb.log({"accuracy_matrix_plot": wandb.Image(fig)})
        plt.close(fig)


def _build_heatmap_figure(matrix_rows, y_labels, tasks, fwt_per_task=None):
    matrix = np.array(matrix_rows, dtype=float)
    num_rows, num_cols = matrix.shape

    has_fwt = fwt_per_task is not None and any(v is not None for v in fwt_per_task)
    if has_fwt:
        fig = plt.figure(figsize=(10, 6), constrained_layout=False)
        gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[14, 0.9, 2.5], wspace=0.20)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        fax = fig.add_subplot(gs[0, 2])
    else:
        fig = plt.figure(figsize=(8, 6), constrained_layout=False)
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[14, 0.9], wspace=0.18)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        fax = None

    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(num_cols))
    ax.set_yticks(range(num_rows))
    ax.set_xticklabels([f"T{i+1}\n{tasks[i]}" for i in range(num_cols)], fontsize=7)
    ax.set_yticklabels(y_labels)

    for i in range(num_rows):
        for j in range(num_cols):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, cax=cax)

    if fax is not None:
        fax.axis("off")
        if fwt_per_task:
            lines = []
            for i, fwt in enumerate(fwt_per_task):
                lines.append(f"After T{i+1}: {fwt:.4f}" if fwt is not None else f"After T{i+1}: N/A")
            fax.text(0.0, 0.5, "FWT\n" + "\n".join(lines), va="center", ha="left", fontsize=9,
                     bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.95, "edgecolor": "lightgray"})

    return fig


def plot_results_matrix(results_matrix, output_dir, cfg, baselines=None, fwt_per_task=None):
    if plt is None or not results_matrix:
        return
    num_eval_tasks = len(results_matrix[0])
    matrix_rows = []
    y_labels = []
    if baselines is not None:
        matrix_rows.append(list(baselines))
        y_labels.append("Zero-shot")
    for i, row in enumerate(results_matrix):
        matrix_rows.append(list(row))
        y_labels.append(f"After T{i+1}")
    os.makedirs(output_dir, exist_ok=True)

    tasks = list(cfg.data.tasks)
    fig = _build_heatmap_figure(matrix_rows, y_labels, tasks, fwt_per_task)

    output_path = os.path.join(output_dir, "trace_accuracy_matrix.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)