import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_hp_comparison(results_df, 
                       metric_cols, 
                       x_col_name='Method',
                       figsize=(12, 6),
                       x_col_fontsize=12,
                       title='Hyperparameter Tuning Comparison',
                       title_fontsize=14,
                       label_fontsize=10,
                       label_color='black',
                       label_position='above',   # Now functional again
                       x_label_rotation=30,
                       show_legend=True,
                       **barplot_kwargs):

    if isinstance(metric_cols, str):
        metric_cols = [metric_cols]

    plot_df = results_df.copy()
    if len(metric_cols) > 1:
        plot_df = pd.melt(plot_df, id_vars=[x_col_name], value_vars=metric_cols,
                          var_name='Metric', value_name='Value')
    else:
        plot_df['Metric'] = metric_cols[0]
        plot_df = plot_df.rename(columns={metric_cols[0]: 'Value'})

    plt.figure(figsize=figsize)

    ax = sns.barplot(data=plot_df, x=x_col_name, y='Value', hue='Metric', 
                     dodge=len(metric_cols) > 1, **barplot_kwargs)

    plt.title(title, fontsize=title_fontsize)
    plt.ylim(0, plot_df['Value'].max() * 1.15)
    plt.xlabel(x_col_name, fontsize=x_col_fontsize)
    plt.ylabel('Value')

    if x_label_rotation is not None:
        plt.xticks(rotation=x_label_rotation, ha='right')

    if label_position.lower() == 'above':
        label_type = 'edge'
        padding = 3
    elif label_position.lower().startswith('cent'):
        label_type = 'center'
        padding = 0
    else:
        raise ValueError("label_position must be either 'above' or 'center'.")

    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', label_type=label_type, padding=padding, fontsize=label_fontsize, color=label_color)

    if show_legend and len(metric_cols) > 1:
        plt.legend(title='Metric')
    else:
        plt.legend([], [], frameon=False)

    plt.tight_layout()
    plt.show()
