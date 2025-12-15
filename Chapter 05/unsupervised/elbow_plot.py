def plot_elbow_method(X, model_class, 
                      max_k=10, figsize=(8, 6), 
                      title='Elbow Method For Optimal K', 
                      xlabel='Number of clusters (K)', 
                      ylabel='Inertia (SSE)', 
                      grid=True, **kwargs):
    inertia = []
    k_val_rng = range(1, max_k + 1)  
    for k in k_val_rng:
        model = model_class(n_clusters=k, **kwargs)
        model.fit(X)
        if hasattr(model, 'inertia_'):
            inertia.append(model.inertia_)
        else:
            print(f"{model_class.__name__} doesn't have 'inertia_' for k={k}. Skipping...")
            break
    if inertia:
        plt.figure(figsize=figsize)
        plt.plot(k_val_rng, inertia, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if grid:
            plt.grid(True)
        plt.show()
