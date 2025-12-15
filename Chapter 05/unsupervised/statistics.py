from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, calinski_harabasz_score

def compute_silhouette(X, model):
    labels = model.labels_ if hasattr(model, 'labels_') else model.fit_predict(X)
    score = silhouette_score(X, labels)
    return score

def compute_davies_bouldin(X, model):
    labels = model.labels_ if hasattr(model, 'labels_') else model.fit_predict(X)
    score = davies_bouldin_score(X, labels)
    return score

def compute_adjusted_rand(X, model, ground_truth_labels):
    labels = model.labels_ if hasattr(model, 'labels_') else model.fit_predict(X)
    score = adjusted_rand_score(ground_truth_labels, labels)
    return score

def compute_calinski_harabasz(X, model):
    labels = model.labels_ if hasattr(model, 'labels_') else model.fit_predict(X)
    score = calinski_harabasz_score(X, labels)
    return score