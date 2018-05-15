import pandas as pd
import networkx as nx

wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
file = ['round1_ijcai_18_train_20180301.txt', 'round1_ijcai_18_test_a_20180301.txt', 'round1_ijcai_18_test_b_20180418.txt']
    
print('loading...')
train = pd.read_csv(wd+file[0], sep=" ")
test_a = pd.read_csv(wd+file[1], sep=" ")
test_b = pd.read_csv(wd+file[2], sep=" ")
data = pd.concat([train, test_a, test_b])

print('graph generating...')
G_ui = nx.from_pandas_edgelist(df=data, source='user_id', target='item_id', edge_attr='is_trade', create_using=nx.MultiGraph())

pagerank = pd.DataFrame(list(nx.pagerank(G_ui).items()), columns=['node', 'pagerank'])

print('merging...')
data = data.merge(pagerank, left_on='user_id', right_on='node', how='left').merge(pagerank, left_on='item_id', right_on='node', how='left')

pagerank_data = pd.DataFrame(columns=['instance_id', 'user_pagerank', 'item_pagerank'])
pagerank_data['instance_id'] = data['instance_id']
pagerank_data['user_pagerank'] = data['pagerank_x']
pagerank_data['item_pagerank'] = data['pagerank_y']

print('saving...')
pagerank_data.to_csv(wd+'pagerank_union.txt', index=False, sep=' ')