#-------- FUNCTIONS TO CLUSTER MODELS --------#
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow import keras

import os
import sys

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


def main() -> None:
    #MODEL_NAME = 'RNN' # BERT,RNN
    MODEL_NAME = sys.argv[1]
    #N_CLUSTERS = 50
    N_CLUSTERS = 50 if len(sys.argv) < 3 else int(sys.argv[2])

    model_path = os.path.join('models', MODEL_NAME)
    model = keras.models.load_model(os.path.join(model_path, 'base_model'))
    clustered_model = cluster_model(model, N_CLUSTERS)
    clustered_model.save(os.path.join('models', f'clustered_{MODEL_NAME}', 'base_model'))


if __name__ == '__main__':
    main()