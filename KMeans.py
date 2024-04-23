import math
import pandas as pd
from data_processing import df, Normalization, SplitData

def Initialization(df):

    random_points = df.sample(n=2)

    return random_points

def Clustering(df, random_points, old_clusters):

    clusters = {i: [] for i in range(len(random_points))}

    for data_num, data_row in df.iterrows():
        data_index = data_row[0]
        data_features = data_row[2:].tolist()
        min_dist = math.inf
        cur_dist = 0
        i = 0
        for centroid_num, centroid_row in random_points.iterrows():
            centroid_index = centroid_row[0]
            centroid_features = centroid_row[2:].tolist()
            cur_dist = Euclidean_Distance(data_features, centroid_features)
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_cluster_index = i
            i += 1
        clusters[min_cluster_index].append(data_index)

    return clusters

# Euclidean_Distance(): Finds the euclidean distance between the training and testing features
def Euclidean_Distance(train_features, test_features):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(train_features, test_features)))

def Means(df, clusters):

    columns = df.columns
    means = pd.DataFrame(columns=columns)

    for key, value in clusters.items():
        means['id'] = key
        feature_dict = {col: 0 for col in df.columns}
        for index in value:
            row = df.loc[df['id'] == index]
            for feature in row:
                if feature != 'id' and feature != 'diagnosis':
                    feature_dict[feature] += df.loc[df['id'] == index, feature].iloc[0]
        vals = list(feature_dict.values())
        if len(value) != 0:
            feature_vals = [val / len(value) for val in vals]
        else:
            feature_vals = [0] * len(feature_dict)
        means.loc[key] = feature_vals

    return means

def main():

    random_points = Initialization(df)

    clusters = {i: [] for i in range(len(random_points))}
    new_clusters = Clustering(df, random_points, clusters)

    while(True):
        means = Means(df, clusters)
        new_clusters = Clustering(df, means, clusters)
        print(new_clusters)
        if clusters == new_clusters:
            break
        else:
            clusters = new_clusters

if __name__ == "__main__":
    main()
