# src/plotting/colors.py
import matplotlib as mpl

def set_paper_style():
    """Set a uniform style for publication-quality plots."""
    
    mpl.rcParams.update({
        # Font sizes
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,

        # Line styles
        "lines.linewidth": 2,
        "lines.markersize": 6,

        # Axes
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",

        # Figure
        "figure.figsize": (6, 4),   # default figure size in inches
        "figure.dpi": 150,

        # Legend
        "legend.frameon": False
    })


# Sky types
SKY_TYPE_COLORS = {
    "clear": "#29e7f8",     
    "mixed": "#db0eff",     
    "overcast": "#2624a7"   
}

# Cloud properties
CLOUD_PROP_COLORS = {
    "cgt": "#07393C",  # midnight green
    "cth": "#90DDF0",  # carribbean blue
    "cot": "#2C666E"   # non photo blue
}

SIMULATION_COLOR = "#35E15A"
OBSERVATION_COLOR = "#520fee"

SIMULATION_LS = ":"
OBSERVATION_LS = "-"

SIMULATION_M = "o"
OBSERVATION_M = "x"


STATION_COLORS = {
    "Florida": "#EF630D",
    "Flesland": "#00aaff"
}

# General plot settings (optional)
DEFAULT_ALPHA = 0.7
DEFAULT_LINESTYLE = "-"
