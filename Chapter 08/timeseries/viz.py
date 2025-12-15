import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import pandas as pd

def plot_time_series(data: pd.DataFrame, 
                     columns: List[str],
                     title: str = "Time Series Plot",
                     xlab:str = "Date", ylab:str = "Values",
                     figsize: tuple = (12,6),
                     colors: List[str] = None, linewidth:float=2.0,
                     max_cols: int = 3, title_fontsize:int=14, 
                     axis_label_fontsize:int=12, plot_grid:bool = True,
                     show_legend=True):

    if colors is None:
        color_pal  = sns.color_palette("husl", len(columns))

    else: 
        color_pal = colors[:len(columns)]
        plt.figure(figsize=figsize)

    for i, col in enumerate(columns[:max_cols]):
        if col in data.columns:
            sns.lineplot(x=data.index, y=data[col],
                            color=color_pal[i], linewidth=linewidth, 
                            label=col)
        else:
            print(f"Warning: Column '{col}' not found in the DataFrame.")

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlab, fontsize=axis_label_fontsize)
    plt.ylabel(ylab, fontsize=axis_label_fontsize)
    plt.legend().set_visible(show_legend)
    plt.grid(plot_grid)
    plt.show()
