import os

def load_parity_plot():
    """
    Returns path to parity plot if exists.
    """
    plot_path = os.path.join("static", "plots", "parity_plot.png")
    if os.path.exists(plot_path):
        return plot_path
    return None
