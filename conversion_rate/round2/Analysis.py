import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def target_variable(target, variable, data):
    sns.pointplot(x=variable, y=target, data=data)
    plt.ylabel(target, fontsize=12)
    plt.xlabel(variable, fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()

def TarVarPlot(data):
	target = 'is_trade'
	drop_list = ['instance_id', 'is_trade',
             	 'item_id', 'item_category_list', 'item_property_list', 'item_brand_id', 'item_city_id',
             	 'user_id', 
             	 'context_id', 'predict_category_property', 'context_timestamp', 'realtime',
              	 'shop_id', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
	plt_list = [x for x in data.columns if x not in drop_list]

	for var in plt_list:
	    print('-'*15, var, '-'*15)
	    target_variable(target, var, data)