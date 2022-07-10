#-------- FUNCTIONS TO CLUSTER MODELS --------#
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow import keras

def cluster_model(model:keras.layers.Layer, n_clusters:int) -> keras.layers.Layer:
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
    'number_of_clusters': n_clusters,
    'cluster_centroids_init': CentroidInitialization.LINEAR
    }
    clustered_model = cluster_weights(model, **clustering_params)
    final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

    return final_model