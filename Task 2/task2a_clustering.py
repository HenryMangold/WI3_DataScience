import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,AffinityPropagation,AgglomerativeClustering,Birch,MiniBatchKMeans,MeanShift, OPTICS,SpectralClustering
    #BisectingKMeans,

class Clustering:

    def run_clustering(self, data_list, hyperparameter_dict):
        df = pd.DataFrame()
        count = 1

        for data in data_list:
            for cluster_type in hyperparameter_dict['cluster_types']:
                if cluster_type == "KMeans":
                    for n_cluster in hyperparameter_dict['n_clusters']:
                        df.loc[count, "wordreduction"] = data['wordreduction']
                        df.loc[count, "grams"] = data['grams']
                        df.loc[count, "matrix"] = data['matrix']
                        df.loc[count,"cluster_type"] = "KMeans"
                        df.loc[count,"n_cluster"] = n_cluster
                        df.loc[count,"labels"] = self.cluster_kmeans(data['data'],n_cluster)
                        count += 1

                elif cluster_type == "DBSCAN":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    df.loc[count,"cluster_type"] = "DBSCAN"
                    df.loc[count,"labels"] = self.cluster_dbscan(data['data'])
                    count += 1

                elif cluster_type == "AffinityPropagation":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    df.loc[count,"cluster_type"] = "AffinityPropagation"
                    df.loc[count,"labels"] = self.cluster_ap(data['data'])
                    count += 1

                elif cluster_type == "AgglomerativeClustering":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    df.loc[count,"cluster_type"] = "AgglomerativeClustering"
                    df.loc[count,"labels"] = self.cluster_ac(data['data'])
                    count += 1

                elif cluster_type == "Birch":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    df.loc[count,"cluster_type"] = "Birch"
                    df.loc[count,"labels"] = self.cluster_birch(data['data'])
                    count += 1

                # Only available in new version of scikitlearn, not compatible
                #elif cluster_type == "BisectingKMeans":
                   # df.loc[count, "wordreduction"] = data['wordreduction']
                   # df.loc[count, "grams"] = data['grams']
                   # df.loc[count, "matrix"] = data['matrix']
                   # df.loc[count,"cluster_type"] = "BisectingKMeans"
                   # df.loc[count,"labels"] = self.cluster_bkmeans(data['data'])
                   # count += 1

                elif cluster_type == "MiniBatchKMeans":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    df.loc[count,"cluster_type"] = "MiniBatchKMeans"
                    df.loc[count,"labels"] = self.cluster_mbkmeans(data['data'])
                    count += 1

                elif cluster_type == "MeanShift":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    df.loc[count,"cluster_type"] = "MeanShift"
                    df.loc[count,"labels"] = self.cluster_ms(data['data'])
                    count += 1

                elif cluster_type == "OPTICS":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    df.loc[count,"cluster_type"] = "OPTICS"
                    df.loc[count,"labels"] = self.cluster_optics(data['data'])
                    count += 1

                elif cluster_type == "SpectralClustering":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    df.loc[count,"cluster_type"] = "SpectralClustering"
                    df.loc[count,"labels"] = self.cluster_sc(data['data'])
                    count += 1

                else:
                    raise ValueError("No valid clustering type selected.")

        return df


    def cluster_kmeans(self, data, n_clusters=8):
        pred = KMeans(n_clusters = n_clusters).fit_predict(data, n_clusters, random_state=5)
        return pred


    def cluster_dbscan(self, data, eps=0.5, min_samples=5):
        pred = DBSCAN().fit_predict(data, eps, min_samples, random_state=5)
        return pred


    def cluster_ap(self, data, damping=0.5, max_iter=200, convergence_iter=15):
        pred = AffinityPropagation().fit_predict(data, damping, max_iter, convergence_iter, random_state=5)
        return pred


    def cluster_ac(self, data, n_clusters=2):
        pred = AgglomerativeClustering().fit_predict(data, n_clusters, random_state=5)
        return pred


    def cluster_birch(self, data, threshold=0.5, branching_factor=50, n_clusters=3):
        pred = Birch().fit_predict(data, threshold, branching_factor, n_clusters, random_state=5)
        return pred

    # Only available in new version of scikitlearn, not compatible
    #def cluster_bkmeans(self, data):
    #    pred = BisectingKMeans().fit_predict(data, random_state=5)
    #    return pred


    def cluster_mbkmeans(self, data, n_clusters=8, max_iter=100, batch_size=1024):
        pred = MiniBatchKMeans().fit_predict(data, n_clusters, max_iter, batch_size, random_state=5)
        return pred


    def cluster_ms(self, data, max_iter=300):
        pred = MeanShift().fit_predict(data, max_iter, random_state=5)
        return pred


    def cluster_optics(self, data, min_samples=5):
        pred = OPTICS().fit_predict(data, min_samples, random_state=5)
        return pred


    def cluster_sc(self, data, n_clusters=3, n_init=10):
        pred = SpectralClustering().fit_predict(data, n_clusters,n_init,random_state=5)
        return pred
