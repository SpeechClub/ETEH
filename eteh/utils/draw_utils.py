import matplotlib.pyplot as plt

def plot_data(data):
    fig = plt.Figure()
    axes = fig.subplots(1)
    axes.imshow(data)
    return fig


def _plot_and_save_attention(att_w):
    # dynamically import matplotlib due to not found error
    from matplotlib.ticker import MaxNLocator
    w, h = plt.figaspect(1.0 / len(att_w))
    fig = plt.Figure(figsize=(w * 2, h * 2))
    axes = fig.subplots(1, len(att_w))
    if len(att_w) == 1:
        axes = [axes]
    for ax, aw in zip(axes, att_w):
        # plt.subplot(1, len(att_w), h)
        ax.imshow(aw, aspect="auto")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def savefig(plot, filename):
    plot.savefig(filename, dpi=300)
    plt.clf()

def draw_matrix(x, exp_file):
    fig_in = plot_data(x)
    savefig(fig_in, exp_file)

