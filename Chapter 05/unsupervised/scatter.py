import matplotlib.pyplot as plt

def plot_scatter(X, labels, title, 
                 title_fsize=14,
                 lab_font_size=12, 
                 cmap='winter', 
                 axis1=0, 
                 axis2=1,
                 scatter_edge_color='k'):
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, axis1], X[:, axis2], 
                          c=labels, cmap=cmap, s=40, 
                          edgecolor=scatter_edge_color, alpha=0.7)
    plt.colorbar(scatter, label='Cluster/Outlier Labels')
    plt.title(title, fontsize=title_fsize)
    plt.xlabel(f"Feature {axis1+1}", fontsize=lab_font_size)
    plt.ylabel(f"Feature {axis2+1}", fontsize=lab_font_size)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()