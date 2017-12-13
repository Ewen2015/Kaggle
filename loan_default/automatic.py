import os 
import pandas as pd 
import numpy as np 


"""
1. non-feature -> target, index
2. duplicated features
3. feature classification: object and numeric
4. correlated features -> difference: Golden Feature Set (GFS) -> remove highly correlated

"""

def corr_pairs(df, gamma=0.99999):
    corr_matrix = df.corr().abs()
    os = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
          .stack()
          .sort_values(ascending=False))
    return os[os>gamma].index.values.tolist()

def features_dup(df, n_head = 5000, print_dup = False):
    """Check first n_head rows and obtain duplicated features."""
    dup_features = []
    if dataset.head(n_head).T.duplicated().any():
        dup_list = np.where(df.head(n_head).T.duplicated())[0].tolist()
        dup_features = df.columns[dup_list]
        if print_dup:
            print(dup_features)
    return dup_features

def features_clf(df, features):
    features = [x for x in features if x not in features_dup(df)]
    dtypes = df[features].dtypes.apply(lambda x: x.name).to_dict()
    int_features, float_features, object_features = [], [], []
    for col, dtype in dtypes.items():
        if dtype == 'int64':
            int_features.append(col)
        elif dtype == 'float64':
            float_features.append(col)
        elif dtype == 'object':
            object_features.append(col)
	return int_features, float_features, object_features



def main():
	path = '/Users/ewenwang/Documents/loan'
	filename = 'train_v2.csv'

	dataset = pd.read_csv(os.path.join(path, filename), low_memory=False)

    target = 'loss'
    dup_features = features_dup(dataset)
    features = [x for x in dataset.columns if x not in [target, 'id']+dup_features.tolist()]
    int_features, float_features, object_features = features_clf(dataset, features)



if __name__ == '__main__':
	main()