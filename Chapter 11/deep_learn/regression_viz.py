import scipy.stats as stats
import matplotlib.pyplot as plt

def dist_of_residuals(y_test, y_pred, 
                      figsize=(6, 6), alpha=0.7,
                      bins=30, hist_col='grey', edge_col='black',
                      line_col='black', line_width=1, line_style='--',
                      x_label='Residuals', y_label='Frequency',
                      title='Distribution of Residuals', 
                      show_grid=True):
    global residuals
    residuals = y_test - y_pred
    plt.figure(figsize=figsize)
    plt.hist(residuals, bins=bins, alpha=alpha, color=hist_col, edgecolor=edge_col)
    plt.axvline(x=0, color=line_col, linestyle=line_style, lw=line_width)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(show_grid)
    plt.show()

def actual_vs_predicted_plot(y_test, y_pred, 
                             figsize=(6, 6), alpha=0.5,
                             point_col='grey', line_col='black',
                             x_label='Actual', y_label='Predicted', 
                             title='Actual vs. Predicted', show_grid=True,
                             line_style='--', line_width=1):
    plt.figure(figsize=figsize)
    plt.scatter(y_test, y_pred, alpha=alpha, color=point_col)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             color=line_col, linestyle=line_style, lw=line_width) 
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(show_grid)
    plt.show()

def qq_plot(residuals, figsize=(6, 6), 
            line_color='red', point_color='blue', 
            line_style='-', point_size=20,
            grid=True, title='Q-Q Plot of Residuals',
            font_size=12):
    plt.figure(figsize=figsize)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

    # Plot the Q-Q line
    plt.plot(osm, slope * np.array(osm) + intercept, line_style, color=line_color, label='Q-Q Line')

    # Plot the actual residuals
    plt.scatter(osm, osr, color=point_color, s=point_size, alpha=0.6, label='Residuals')

    plt.title(title, fontsize=font_size + 2)
    plt.xlabel("Theoretical Quantiles", fontsize=font_size)
    plt.ylabel("Ordered Values", fontsize=font_size)
    if grid:
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
