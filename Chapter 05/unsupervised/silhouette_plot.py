from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def silhouette_plot(X, labels, n_clusters, 
                    title="Silhouette Plot for Clustering",
                    fig_size=(8,6),
                    x_lab='Coefficient Values',
                    y_lab='Cluster',
                    silh_colors=None,
                    x_axis_line_col='black'):

    sil_vals = silhouette_samples(X, labels)
    mean_score = silhouette_score(X, labels)

    _, ax = plt.subplots(figsize=fig_size)
    y_lower = 10

    # Use default colours if none are provided
    if silh_colors is None:
        silh_colors = plt.cm.tab10.colors  # Default colour palette

    for i in range(n_clusters):
        cluster_sil_vals = sil_vals[labels == i]
        cluster_sil_vals.sort()
        y_upper = y_lower + len(cluster_sil_vals)

        colour = silh_colors[i % len(silh_colors)]  

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_sil_vals, alpha=0.7, color=colour
        )
        ax.text(-0.05, y_lower + 0.5 * len(cluster_sil_vals), str(i))
        
        y_lower = y_upper + 10

    ax.axvline(x=mean_score, color=x_axis_line_col, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_yticks([]) 
    grid_lines = np.arange(-0.1, 1.1, 0.1)
    plt.show()
