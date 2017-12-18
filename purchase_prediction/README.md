## JData Competition

### Training Data Setup

#### Introduction

We considered three ways to build up a training dataset, 

- Start with `user_id`: firstly, predict who would buy anything, and then predict what would they buy.
    
- Start with `sku_id`: consider each product as an object, and predict if it would be bought and by whom.
    
- Start with `user_id and sku_id` pair: take these pairs as objects and check if the any purchase activity would happen in 5 days.
    

We finally decided to start with the third method and may consider other ways when time allows.

#### User and Item Pair

Features may be considered:

- **User**: age, sex, level, user_lv_cd, browse_num, addcart_num, delcart_num, buy_num, favor_num, click_num
- **Item**: attr1, attr2, attr3, cate, brand, comment_num, has_bad_comment, bad_comment_rate, browse_num, addcart_num, delcart_num, buy_num, favor_num, click_num
- **Behavior**: brow_num_today, addcart_num_today, delcart_num_today, buy_num_today, favor_num_today, click_num_today; brow_num_past, addcart_num_past, delcart_num_past, buy_num_past, favor_num_past, click_num_past;

In particular, the first training data set takes behavior of today (4/10), behavior of past 5 days (4/5 - 4/9), user_table, and item_table as features. The dependent variable is whether any purchase happeds from 4/11 to 4/15. 

#### Discussion

The training data naturaly blocks the user w/o records in today and past 5 days, which can low down the precise of predicting. 

---

### 任务描述：

参赛者需要使用京东多个品类下商品的历史销售数据，构建算法模型，预测用户在未来5天内，对某个目标品类下商品的购买意向。对于训练集中出现的每一个用户，参赛者的模型需要预测该用户在未来5天内是否购买目标品类下的商品以及所购买商品的SKU_ID。评测算法将针对参赛者提交的预测结果，计算加权得分。

### Scoring

参赛者提交的结果文件中包含对所有用户购买意向的预测结果。对每一个用户的预测结果包括两方面：

- 1、该用户2016-04-16到2016-04-20是否下单P中的商品，提交的结果文件中仅包含预测为下单的用户，预测为未下单的用户，无须在结果中出现。若预测正确，则评测算法中置label=1，不正确label=0；
- 2、如果下单，下单的sku_id （只需提交一个sku_id），若sku_id预测正确，则评测算法中置pred=1，不正确pred=0。

对于参赛者提交的结果文件，按如下公式计算得分：

$$Score=0.4*F11 + 0.6* F12$$

此处的F1值定义为：

$$F11=6*Recall*Precise/(5*Recall+Precise)$$
$$F12=5*Recall*Precise/(2*Recall+3*Precise)$$

其中，Precise为准确率，Recall为召回率.

F11是label=1或0的F1值，F12是pred=1或0的F1值.
