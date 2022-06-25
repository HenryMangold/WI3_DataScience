import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,AffinityPropagation,AgglomerativeClustering,Birch,MiniBatchKMeans,MeanShift, OPTICS,SpectralClustering
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
    #BisectingKMeans,


class Clustering:
    """ Class which implements the clustering algorithms

    Methods:
        find_optimal_k(start, end, data_list)
            Finding the optimal value for K with help of elbow and silhouette method
        run_clustering(data_list, hyperparameter_dict)
            Runs clustering based on give hyperparameter set
        cluster_kmeans(data, n_clusters)
            Clustering based on KMeans algorithm
        cluster_birch(data, threshold, branching_factor)
            Clustering based on Birch algorithm
        cluster_ac(data, n_clusters)
            Clustering based on AgglomerativeClustering algorithm
        cluster_ap(data, damping)
            Clustering based on AffinityPropagation algorithm
        cluster_dbscan(data, eps)
            Clustering based on DBSCAN algorithm
        cluster_sc(self, data, n_clusters)
            Clustering based on SpectralClustering algorithm
        cluster_optics(data)
            Clustering based on OPTICS algorithm
        cluster_ms(data)
            Clustering based on MiniBatchKMeans algorithm
        cluster_mbkmeans(data, n_clusters)
            Clustering based on MeanShift algorithm

    """


    def find_optimal_k(self,start,end, data_list):
        """ Finding the optimal value for K with help of elbow and silhouette method

        Args:
            start: min value for K
            end: max value for K
            data_list: data to test

        Returns:
            List -> Containing the results for every dataset as dict with keys: distortions,
                                                                                inertias,
                                                                                silhouette_scores,
                                                                                silhouette_values,
                                                                                labels,
                                                                                wordreduction,
                                                                                grams,
                                                                                matrix

        """

        K = range(start, end)

        results = []

        for data in data_list:
            X = data['data'].drop('Place', axis='columns')
            distortions = []
            inertias = []
            silhouette_scores = []
            silhouette_values = []
            labels = []

            for k in K:
                kmeanModel = KMeans(n_clusters=k, random_state=5).fit(X)
                kmeanModel.fit(X)

                distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                                    'euclidean'), axis=1)) / X.shape[0])
                inertias.append(kmeanModel.inertia_)

                silhouette_scores.append(silhouette_score(X, kmeanModel.labels_))

                silhouette_values.append(silhouette_samples(X, kmeanModel.labels_))

                labels.append(kmeanModel.labels_)

            results.append({'distortions': distortions,
                            'inertias': inertias,
                            'silhouette_scores': silhouette_scores,
                            'silhouette_values': silhouette_values,
                            'labels': labels,
                            'wordreduction': data['wordreduction'],
                            'grams': data['grams'],
                            'matrix': data['matrix']})

        return K, results


    def run_clustering(self, data_list, hyperparameter_dict):
        """ Runs clustering based on give hyperparameter set

        Args:
            data_list: data to train on
            hyperparameter_dict: hyperparameters for training

        Returns:
            DataFrame -> Containing training params and results

        """

        df = pd.DataFrame()
        count = 1
        column_labels = []

        for data in data_list:
            path = f"output/{data['wordreduction']}_{data['grams']}_{data['matrix']}.csv"
            data['data'].to_csv(path)

            for cluster_type in hyperparameter_dict['cluster_types']:
                if cluster_type == "KMeans":
                    for n_cluster in hyperparameter_dict['n_clusters']:
                        df.loc[count, "wordreduction"] = data['wordreduction']
                        df.loc[count, "grams"] = data['grams']
                        df.loc[count, "matrix"] = data['matrix']
                        df.loc[count, "n_cluster"] = n_cluster
                        column_labels.append(list(self.cluster_kmeans(data['data'], n_cluster)))
                        df.loc[count, "train_data"] = path
                        df.loc[count, "cluster_type"] = cluster_type
                        count += 1

                elif cluster_type == "DBSCAN":
                    for eps in hyperparameter_dict['eps']:
                        df.loc[count, "wordreduction"] = data['wordreduction']
                        df.loc[count, "grams"] = data['grams']
                        df.loc[count, "matrix"] = data['matrix']
                        df.loc[count, "eps"] = eps
                        column_labels.append(list(self.cluster_dbscan(data['data'], eps)))
                        df.loc[count, "train_data"] = path
                        df.loc[count, "cluster_type"] = cluster_type
                        count += 1

                elif cluster_type == "AffinityPropagation":
                    for damping in hyperparameter_dict['damping']:
                        df.loc[count, "wordreduction"] = data['wordreduction']
                        df.loc[count, "grams"] = data['grams']
                        df.loc[count, "matrix"] = data['matrix']
                        df.loc[count, "damping"] = damping
                        column_labels.append(list(self.cluster_ap(data['data'], damping)))
                        df.loc[count, "train_data"] = path
                        df.loc[count, "cluster_type"] = cluster_type
                        count += 1

                elif cluster_type == "AgglomerativeClustering":
                    for n_cluster in hyperparameter_dict['n_clusters']:
                        df.loc[count, "wordreduction"] = data['wordreduction']
                        df.loc[count, "grams"] = data['grams']
                        df.loc[count, "matrix"] = data['matrix']
                        df.loc[count, "n_cluster"] = n_cluster
                        column_labels.append(list(self.cluster_ac(data['data'], n_cluster)))
                        df.loc[count, "train_data"] = path
                        df.loc[count, "cluster_type"] = cluster_type
                        count += 1

                elif cluster_type == "Birch":
                    for threshold in hyperparameter_dict['threshold']:
                        for branching_factor in hyperparameter_dict['branching_factor']:
                            df.loc[count, "wordreduction"] = data['wordreduction']
                            df.loc[count, "grams"] = data['grams']
                            df.loc[count, "matrix"] = data['matrix']
                            df.loc[count, "threshold"] = threshold
                            df.loc[count, "branching_factor"] = branching_factor
                            column_labels.append(list(self.cluster_birch(data['data'], threshold, branching_factor)))
                            df.loc[count, "train_data"] = path
                            df.loc[count, "cluster_type"] = cluster_type
                            count += 1

                # Only available in new version of scikitlearn, not compatible
                # elif cluster_type == "BisectingKMeans":
                # df.loc[count, "wordreduction"] = data['wordreduction']
                # df.loc[count, "grams"] = data['grams']
                # df.loc[count, "matrix"] = data['matrix']
                # df.loc[count,"cluster_type"] = "BisectingKMeans"
                # df.loc[count,"labels"] = self.cluster_bkmeans(data['data'])
                # count += 1

                elif cluster_type == "MiniBatchKMeans":
                    for n_cluster in hyperparameter_dict['n_clusters']:
                        df.loc[count, "wordreduction"] = data['wordreduction']
                        df.loc[count, "grams"] = data['grams']
                        df.loc[count, "matrix"] = data['matrix']
                        df.loc[count, "n_cluster"] = n_cluster
                        column_labels.append(list(self.cluster_mbkmeans(data['data'], n_cluster)))
                        df.loc[count, "train_data"] = path
                        df.loc[count, "cluster_type"] = cluster_type
                        count += 1

                elif cluster_type == "MeanShift":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    column_labels.append(list(self.cluster_ms(data['data'])))
                    df.loc[count, "train_data"] = path
                    df.loc[count, "cluster_type"] = cluster_type
                    count += 1

                elif cluster_type == "OPTICS":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    column_labels.append(list(self.cluster_optics(data['data'])))
                    df.loc[count, "train_data"] = path
                    df.loc[count, "cluster_type"] = cluster_type
                    count += 1

                elif cluster_type == "SpectralClustering":
                    for n_cluster in hyperparameter_dict['n_clusters']:
                        df.loc[count, "wordreduction"] = data['wordreduction']
                        df.loc[count, "grams"] = data['grams']
                        df.loc[count, "matrix"] = data['matrix']
                        df.loc[count, "n_cluster"] = n_cluster
                        column_labels.append(list(self.cluster_sc(data['data'], n_cluster)))
                        df.loc[count, "train_data"] = path
                        df.loc[count, "cluster_type"] = cluster_type
                        count += 1

                else:
                    print(cluster_type)
                    raise ValueError("No valid clustering type selected.")

        df['labels'] = column_labels

        return df


    def cluster_kmeans(self, data, n_clusters=8):
        """ Clustering based on KMeans algorithm

        Args:
            data: data to train on
            n_clusters: amount of clusters

        Returns:
            List -> Predicted labels for clusters

        """
        pred = KMeans(n_clusters=n_clusters, random_state=5).fit_predict(data.drop('Place', axis='columns'))
        return pred


    def cluster_dbscan(self, data, eps=0.5):
        """ Clustering based on DBSCAN algorithm

        Args:
            data: data to train on
            eps: maximum distance between two samples for one to be considered as in the neighborhood of the other

        Returns:
            List -> Predicted labels for clusters

        """
        pred = DBSCAN(eps=eps).fit_predict(data.drop('Place', axis='columns'))
        return pred


    def cluster_ap(self, data, damping=0.5):
        """ Clustering based on AffinityPropagation algorithm

        Args:
            data: data to train on
            damping: damping factor in the range [0.5, 1.0) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping)

        Returns:
            List -> Predicted labels for clusters

        """
        pred = AffinityPropagation(damping=damping, random_state=5).fit_predict(data.drop('Place', axis='columns'))
        return pred


    def cluster_ac(self, data, n_clusters=2):
        """ Clustering based on AgglomerativeClustering algorithm

        Args:
            data: data to train on
            n_clusters: amount of clusters

        Returns:
            List -> Predicted labels for clusters

        """
        pred = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(data.drop('Place', axis='columns'))
        return pred


    def cluster_birch(self, data, threshold=0.5, branching_factor=50):
        """ Clustering based on Birch algorithm

        Args:
            data: data to train on
            threshold: radius of the subcluster obtained by merging a new sample and the closest subcluster
            branching_factor: maximum number of CF subclusters in each node

        Returns:
            List -> Predicted labels for clusters

        """
        pred = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None).fit_predict(
            data.drop('Place', axis='columns'))
        return pred


    # Only available in new version of scikitlearn, not compatible
    # def cluster_bkmeans(self, data):
    #    pred = BisectingKMeans().fit_predict(data, random_state=5)
    #    return pred


    def cluster_mbkmeans(self, data, n_clusters=8):
        """ Clustering based on MiniBatchKMeans algorithm

        Args:
            data: data to train on
            n_clusters: amount of clusters

        Returns:
            List -> Predicted labels for clusters

        """
        pred = MiniBatchKMeans(n_clusters=n_clusters, random_state=5).fit_predict(data.drop('Place', axis='columns'))
        return pred


    def cluster_ms(self, data):
        """ Clustering based on MeanShift algorithm

        Args:
            data: data to train on

        Returns:
            List -> Predicted labels for clusters

        """
        pred = MeanShift().fit_predict(data.drop('Place', axis='columns'))
        return pred


    def cluster_optics(self, data):
        """ Clustering based on OPTICS algorithm

        Args:
            data: data to train on

        Returns:
            List -> Predicted labels for clusters

        """
        pred = OPTICS().fit_predict(data.drop('Place', axis='columns'))
        return pred


    def cluster_sc(self, data, n_clusters=3):
        """ Clustering based on SpectralClustering algorithm

        Args:
            data: data to train on
            n_clusters: amount of clusters

        Returns:
            List -> Predicted labels for clusters

        """
        pred = SpectralClustering(n_clusters=n_clusters, random_state=5).fit_predict(data.drop('Place', axis='columns'))
        return pred
