import pandas as pd
import networkx as nx

def PageRank(data):
	print('graph generating...')
	G_ui = nx.from_pandas_edgelist(df=data, source='user_id', target='item_id', edge_attr=False)
	pagerank = pd.DataFrame(list(nx.pagerank(G_ui).items()), columns=['node', 'pagerank'])

	print('merging...')
	data = data.merge(pagerank, left_on='user_id', right_on='node', how='left').merge(pagerank, left_on='item_id', right_on='node', how='left')

	pagerank_data = pd.DataFrame(columns=['instance_id', 'user_pagerank', 'item_pagerank'])
	pagerank_data['instance_id'] = data['instance_id']
	pagerank_data['user_pagerank'] = data['pagerank_x']
	pagerank_data['item_pagerank'] = data['pagerank_y']

	return pagerank_data

if __name__ == '__main__':
	wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
	file = ['round2_page_train_7.txt', 'round2_page_test_a.txt', 'round2_page_test_b.txt']
	out_file = ['round2_page_train_7.txt', 'round2_page_test.txt']
	print('loading...')
	train = pd.read_csv(wd+file[0], sep=' ')
	test_a = pd.read_csv(wd+file[1], sep=' ')
	test_b = pd.read_csv(wd+file[2], sep=' ')

	test = pd.concat([test_a, test_b])
	pagerank_train = PageRank(train)
	pagerank_test = PageRank(test)
	
	print('saving...')
	pagerank_train.to_csv(wd+out_file[0], index=False, sep=' ')
	pagerank_train.to_csv(wd+out_file[1], index=False, sep=' ')
