import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,AffinityPropagation,AgglomerativeClustering,Birch,BisectingKMeans,MiniBatchKMeans,MeanShift, OPTICS,SpectralClustering

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

                elif cluster_type == "BisectingKMeans":
                    df.loc[count, "wordreduction"] = data['wordreduction']
                    df.loc[count, "grams"] = data['grams']
                    df.loc[count, "matrix"] = data['matrix']
                    df.loc[count,"cluster_type"] = "BisectingKMeans"
                    df.loc[count,"labels"] = self.cluster_bkmeans(data['data'])
                    count += 1

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


    def cluster_kmeans(self, data, n_clusters):
        pred = KMeans(n_clusters = n_clusters).fit_predict(data)
        return pred


    def cluster_dbscan(self, data):
        pred = DBSCAN().fit_predict(data)
        return pred


    def cluster_ap(self, data):
        pred = AffinityPropagation().fit_predict(data)
        return pred


    def cluster_ac(self, data):
        pred = AgglomerativeClustering().fit_predict(data)
        return pred


    def cluster_birch(self, data):
        pred = Birch().fit_predict(data)
        return pred


    def cluster_bkmeans(self, data):
        pred = BisectingKMeans().fit_predict(data)
        return pred


    def cluster_mbkmeans(self, data):
        pred = MiniBatchKMeans().fit_predict(data)
        return pred


    def cluster_ms(self, data):
        pred = MeanShift().fit_predict(data)
        return pred


    def cluster_optics(self, data):
        pred = OPTICS().fit_predict(data)
        return pred


    def cluster_sc(self, data):
        pred = SpectralClustering().fit_predict(data)
        return pred
