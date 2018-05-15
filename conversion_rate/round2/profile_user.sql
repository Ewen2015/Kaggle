-- Project: Conversion Rate Prediction
-- Date: May 8, 2018
-- Author: Ewen

-- ================================
-- seperate training data by day 7
-- create tables

create table summer.round2_train_7 as
select * 
from summer.round2_train
where context_day = 7;

-- ================================
-- Append test_a to train_7 as round2_data_7_a


-- 杜老板，这一步麻烦把 test_a 加在 train_7 后面， 生产一个更大的表 round2_data_7_a

-- test_a 比 train_7 少一列 is_trade, 其他字段都一样

create table as round2_data_7_a as
select `(?!is_trade).+`，is_trade from round2_train_7 
UNION ALL
select *，NULL as is_trade from round2_test_a;


-- ================================
-- feature engineering
-- user profile

-- ============= 
-- user profile 
-- gender 
create table Summer.profile_user_gender as 
select 
user_gender_id, 
AVG(item_price_level) as user_gender_price_avg,
STDDEV(item_price_level)+0.001 as user_gender_price_sd,
AVG(item_sales_level) as user_gender_sales_avg,
STDDEV(item_sales_level)+0.001 as user_gender_sales_sd,
AVG(item_collected_level) as user_gender_collected_avg,
STDDEV(item_collected_level)+0.001 as user_gender_collected_sd,
AVG(item_pv_level) as user_gender_pv_avg,
STDDEV(item_pv_level)+0.001 as user_gender_pv_sd,

AVG(shop_review_num_level) as user_gender_review_num_avg,
STDDEV(shop_review_num_level)+0.001 as user_gender_review_num_sd,
AVG(shop_review_positive_rate) as user_gender_review_positive_avg,
STDDEV(shop_review_positive_rate)+0.001 as user_gender_review_positive_sd,
AVG(shop_star_level) as user_gender_star_avg,
STDDEV(shop_star_level)+0.001 as user_gender_star_sd,
AVG(shop_score_service) as user_gender_service_avg,
STDDEV(shop_score_service)+0.001 as user_gender_service_sd,
AVG(shop_score_delivery) as user_gender_delivery_avg,
STDDEV(shop_score_delivery)+0.001 as user_gender_delivery_sd,
AVG(shop_score_description) as user_gender_description_avg,
STDDEV(shop_score_description)+0.001 as user_gender_description_sd
from Summer.round2_data_7_a
group by user_gender_id;

-- age
create table Summer.profile_user_age as 
select 
user_age_level, 
AVG(item_price_level) as user_age_price_avg,
STDDEV(item_price_level)+0.001 as user_age_price_sd,
AVG(item_sales_level) as user_age_sales_avg,
STDDEV(item_sales_level)+0.001 as user_age_sales_sd,
AVG(item_collected_level) as user_age_collected_avg,
STDDEV(item_collected_level)+0.001 as user_age_collected_sd,
AVG(item_pv_level) as user_age_pv_avg,
STDDEV(item_pv_level)+0.001 as user_age_pv_sd,

AVG(shop_review_num_level) as user_age_review_num_avg,
STDDEV(shop_review_num_level)+0.001 as user_age_review_num_sd, 
AVG(shop_review_positive_rate) as user_age_review_positive_avg,
STDDEV(shop_review_positive_rate)+0.001 as user_age_review_positive_sd,
AVG(shop_star_level) as user_age_star_avg,
STDDEV(shop_star_level)+0.001 as user_age_star_sd,
AVG(shop_score_service) as user_age_service_avg,
STDDEV(shop_score_service)+0.001 as user_age_service_sd,
AVG(shop_score_delivery) as user_age_delivery_avg,
STDDEV(shop_score_delivery)+0.001 as user_age_delivery_sd,
AVG(shop_score_description) as user_age_description_avg,
STDDEV(shop_score_description)+0.001 as user_age_description_sd
from Summer.round2_data_7_a
group by user_age_level;

-- occupation
create table Summer.profile_user_occup as 
select 
user_occupation_id, 
AVG(item_price_level) as user_occupation_price_avg,
STDDEV(item_price_level)+0.001 as user_occupation_price_sd,
AVG(item_sales_level) as user_occupation_sales_avg,
STDDEV(item_sales_level)+0.001 as user_occupation_sales_sd,
AVG(item_collected_level) as user_occupation_collected_avg,
STDDEV(item_collected_level)+0.001 as user_occupation_collected_sd,
AVG(item_pv_level) as user_occupation_pv_avg,
STDDEV(item_pv_level)+0.001 as user_occupation_pv_sd,

AVG(shop_review_num_level) as user_occupation_review_num_avg,
STDDEV(shop_review_num_level)+0.001 as user_occupation_review_num_sd,
AVG(shop_review_positive_rate) as user_occupation_review_postive_avg,
STDDEV(shop_review_positive_rate)+0.001 as user_occupation_review_postive_sd,
AVG(shop_star_level) as user_occupation_star_avg,
STDDEV(shop_star_level)+0.001 as user_occupation_star_sd,
AVG(shop_score_service) as user_occupation_service_avg,
STDDEV(shop_score_service)+0.001 as user_occupation_service_sd,
AVG(shop_score_delivery) as user_occupation_delivery_avg,
STDDEV(shop_score_delivery)+0.001 as user_occupation_delivery_sd,
AVG(shop_score_description) as user_occupation_description_avg,
STDDEV(shop_score_description)+0.001 as user_occupation_description_sd
from Summer.round2_data_7_a
group by user_occupation_id;

-- star
create table Summer.profile_user_star as 
select 
user_star_level, 
AVG(item_price_level) as user_star_price_avg,
STDDEV(item_price_level)+0.001 as user_star_price_sd,
AVG(item_sales_level) as user_star_sales_avg,
STDDEV(item_sales_level)+0.001 as user_star_sales_sd,
AVG(item_collected_level) as user_star_collected_avg,
STDDEV(item_collected_level)+0.001 as user_star_collected_sd,
AVG(item_pv_level) as user_star_pv_avg,
STDDEV(item_pv_level)+0.001 as user_star_pv_sd,

AVG(shop_review_num_level) as user_star_review_num_avg,
STDDEV(shop_review_num_level)+0.001 as user_star_review_num_sd,
AVG(shop_review_positive_rate) as user_star_review_positive_avg,
STDDEV(shop_review_positive_rate)+0.001 as user_star_review_positive_sd,
AVG(shop_star_level) as user_star_star_avg,
STDDEV(shop_star_level)+0.001 as user_star_star_sd,
AVG(shop_score_service) as user_star_service_avg,
STDDEV(shop_score_service)+0.001 as user_star_service_sd,
AVG(shop_score_delivery) as user_star_delivery_avg,
STDDEV(shop_score_delivery)+0.001 as user_star_delivery_sd,
AVG(shop_score_description) as user_star_description_avg,
STDDEV(shop_score_description)+0.001 as user_star_description_sd
from Summer.round2_data_7_a
group by user_star_level;

-- ============= 
-- merge tables
create table Summer.profile_user as 
select 
t1.instance_id, 

(t1.item_price_level - t2.user_gender_price_avg)/t2.user_gender_price_sd as user_gender_price,
(t1.item_price_level - t3.user_age_price_avg)/t3.user_age_price_sd as user_age_price,
(t1.item_price_level - t4.user_occupation_price_avg)/t4.user_occupation_price_sd as user_occupation_price,
(t1.item_price_level - t5.user_star_price_avg)/t5.user_star_price_sd as user_star_price,

(t1.item_sales_level - t2.user_gender_sales_avg)/t2.user_gender_sales_sd as user_gender_sales,
(t1.item_sales_level - t3.user_age_sales_avg)/t3.user_age_sales_sd as user_age_sales,
(t1.item_sales_level - t4.user_occupation_sales_avg)/t4.user_occupation_sales_sd as user_occupation_sales,
(t1.item_sales_level - t5.user_star_sales_avg)/t5.user_star_sales_sd as user_star_sales,

(t1.item_collected_level - t2.user_gender_collected_avg)/t2.user_gender_collected_sd as user_gender_collected,
(t1.item_collected_level - t3.user_age_collected_avg)/t3.user_age_collected_sd as user_age_collected,
(t1.item_collected_level - t4.user_occupation_collected_avg)/t4.user_occupation_collected_sd as user_occupation_collected,
(t1.item_collected_level - t5.user_star_collected_avg)/t5.user_star_collected_sd as user_star_collected,

(t1.item_pv_level - t2.user_gender_pv_avg)/t2.user_gender_pv_sd as user_gender_pv,
(t1.item_pv_level - t3.user_age_pv_avg)/t3.user_age_pv_sd as user_age_pv,
(t1.item_pv_level - t4.user_occupation_pv_avg)/t4.user_occupation_pv_sd as user_occupation_pv,
(t1.item_pv_level - t5.user_star_pv_avg)/t5.user_star_pv_sd as user_star_pv,

(t1.shop_review_num_level - t2.user_gender_review_num_avg)/t2.user_gender_review_num_sd as user_gender_review_num, 
(t1.shop_review_num_level - t3.user_age_review_num_avg)/t3.user_age_review_num_sd as user_age_review_num, 
(t1.shop_review_num_level - t4.user_occupation_review_num_avg)/t4.user_occupation_review_num_sd as user_occupation_review_num, 
(t1.shop_review_num_level - t5.user_star_review_num_avg)/t5.user_star_review_num_sd as user_star_review_num, 

(t1.shop_review_positive_rate - t2.user_gender_review_positive_avg)/t2.user_gender_review_positive_sd as user_gender_review_positive,
(t1.shop_review_positive_rate - t3.user_age_review_positive_avg)/t3.user_age_review_positive_sd as user_age_review_positive,
(t1.shop_review_positive_rate - t4.user_occupation_review_postive_avg)/t4.user_occupation_review_postive_sd as user_occupation_review_postive,
(t1.shop_review_positive_rate - t5.user_star_review_positive_avg)/t5.user_star_review_positive_sd as user_star_review_positive,

(t1.shop_star_level - t2.user_gender_star_avg)/t2.user_gender_star_sd as user_gender_star,
(t1.shop_star_level - t3.user_age_star_avg)/t3.user_age_star_sd as user_age_star,
(t1.shop_star_level - t4.user_occupation_star_avg)/t4.user_occupation_star_sd as user_occupation_star,
(t1.shop_star_level - t5.user_star_star_avg)/t5.user_star_star_sd as user_star_star,

(t1.shop_score_service - t2.user_gender_service_avg)/t2.user_gender_service_sd as user_gender_service,
(t1.shop_score_service - t3.user_age_service_avg)/t3.user_age_service_sd as user_age_service,
(t1.shop_score_service - t4.user_occupation_service_avg)/t4.user_occupation_service_sd as user_occupation_service,
(t1.shop_score_service - t5.user_star_service_avg)/t5.user_star_service_sd as user_star_service,

(t1.shop_score_delivery - t2.user_gender_delivery_avg)/t2.user_gender_delivery_sd as user_gender_delivery, 
(t1.shop_score_delivery - t3.user_age_delivery_avg)/t3.user_age_delivery_sd as user_age_delivery,
(t1.shop_score_delivery - t4.user_occupation_delivery_avg)/t4.user_occupation_delivery_sd as user_occupation_delivery,
(t1.shop_score_delivery - t5.user_star_delivery_avg)/t5.user_star_delivery_sd as user_star_delivery,

(t1.shop_score_description - t2.user_gender_description_avg)/t2.user_gender_delivery_sd as user_gender_description, 
(t1.shop_score_description - t3.user_age_description_avg)/t3.user_age_description_sd as user_age_description, 
(t1.shop_score_description - t4.user_occupation_description_avg)/t4.user_occupation_description_sd as user_occupation_description, 
(t1.shop_score_description - t5.user_star_description_avg)/t5.user_star_description_sd as user_star_description

from round2_data_7_a t1 
left join profile_user_gender t2 on t1.user_gender_id = t2.user_gender_id
left join profile_user_age t3 on t1.user_age_level = t3.user_age_level
left join profile_user_occup t4 on t1.user_occupation_id = t4.user_occupation_id
left join profile_user_star t5 on t1.user_star_level = t5.user_star_level
;



-- 杜老板，我只需要最后这个表 profile_user
