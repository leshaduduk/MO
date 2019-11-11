import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def assign_clusters(samples, centroids):
    distances = euclidean_distances(samples, centroids)
    return pd.DataFrame(distances).idxmin(axis="columns")


def cost(samples, clusters, centroids):
    centroids_for_samples = centroids.iloc[clusters]
    centroids_for_samples.reset_index(drop=True, inplace=True)
    norms = np.linalg.norm(samples - centroids_for_samples, axis=1) ** 2
    return 1 / len(clusters) * norms.sum()


def get_centroids(samples, k):
    return samples.sample(k)


def kmeans(samples, k):
    centroids = get_centroids(samples, k)
    while True:
        clusters = assign_clusters(samples, centroids)
        yield clusters, centroids
        new_centroids = move_centroids(samples, clusters)
        if np.equal(centroids, new_centroids).all().all():
            break
        centroids = new_centroids


def kmeans_multi(samples, k, n):
    best_cost = float("inf")
    best_results = None
    for _ in range(n):
        results = list(kmeans(samples, k))
        clusters, centroids = results[-1]
        current_cost = cost(samples, clusters, centroids)
        if current_cost < best_cost:
            best_cost = current_cost
            best_results = results
    return best_results


def mass_center(cluster):
    return cluster.mean()


def move_centroids(samples, clusters):
    k = clusters.max() + 1
    result = []
    for i in range(k):
        index = clusters[clusters == i].index
        cluster = samples.iloc[index]
        result.append(mass_center(cluster))
    return pd.DataFrame(result)
