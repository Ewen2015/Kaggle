-- Tasks done:
-- 1. created 4 tables: user_table, item_table, shop_table, and user_item_table
-- 2. wrote a SQL to join them
-- Tasks to do:
-- 1. to consider time: like the conversion rate before each records on different perspectives, user, item ...
-- 2. to consider item_category_list: to split it and include them into table generation
-- 3. to consider item_property_list and predict_category_property, maybe in Python
-- 4. to consider more features servicing for the '11.11' shopping festival...

-- user table
CREATE TABLE user_table AS SELECT
user_id,
COUNT( * ) AS user_instance_cnt,
COUNT( DISTINCT item_id ) AS user_item_cnt,
COUNT( * ) / COUNT( DISTINCT item_id ) AS user_item_avg,
COUNT( DISTINCT item_brand_id ) AS user_brand_cnt,
COUNT( DISTINCT item_city_id ) AS user_city_cnt,
AVG( item_price_level ) AS user_price_avg,
STDDEV( item_price_level ) AS user_price_sd,
AVG( item_sales_level ) AS user_sales_avg,
STDDEV( item_sales_level ) AS user_sales_sd,
AVG( item_collected_level ) AS user_collected_avg,
STDDEV( item_collected_level ) AS user_collected_sd,
AVG( item_pv_level ) AS user_pv_avg,
STDDEV( item_pv_level ) AS user_pv_sd,
COUNT( DISTINCT context_id ) AS user_context_cnt,
COUNT( DISTINCT context_page_id ) AS user_context_page_cnt,
COUNT( DISTINCT context_hour ) user_context_hour_cnt,
COUNT( DISTINCT shop_id ) AS user_shop_cnt,
AVG( shop_review_num_level ) AS user_shop_review_num_avg,
STDDEV( shop_review_num_level ) AS user_shop_review_num_sd,
AVG( shop_review_positive_rate ) AS user_shop_review_positive_rate_avg,
STDDEV( shop_review_positive_rate ) AS user_shop_review_positive_rate_sd,
AVG( shop_star_level ) AS user_shop_star_avg,
STDDEV( shop_star_level ) AS user_shop_star_sd,
AVG( shop_score_service ) AS user_shop_score_service_avg,
STDDEV( shop_score_service ) AS user_shop_score_service_sd,
AVG( shop_score_delivery ) AS user_shop_score_delivery_avg,
STDDEV( shop_score_delivery ) AS user_shop_score_delivery_sd,
AVG( shop_score_description ) AS user_shop_score_description_avg,
STDDEV( shop_score_description ) AS user_shop_score_description_sd 
FROM
	Summer.data24_raw 
GROUP BY
	user_id;
	
-- item table
CREATE TABLE item_table AS SELECT
item_id,
COUNT( * ) AS item_instance_cnt,
COUNT( DISTINCT user_id ) AS item_user_cnt,
COUNT( * ) / COUNT( DISTINCT user_id ) AS item_user_avg,
AVG( user_gender_id ) AS item_user_gender_avg,
AVG( user_age_level ) AS item_user_age_avg,
AVG( user_occupation_id ) AS item_user_oppcupation_avg,
AVG( user_star_level ) AS item_user_star_avg,
COUNT( DISTINCT context_id ) AS item_context_cnt,
COUNT( DISTINCT context_page_id ) AS item_context_page_cnt,
COUNT( DISTINCT context_hour ) AS item_context_hour_cnt,
AVG( context_hour ) AS item_context_hour_avg 
FROM
	Summer.data24_raw 
GROUP BY
	item_id;
	
-- shop table
CREATE TABLE shop_table AS SELECT
shop_id,
COUNT( * ) AS shop_instance_cnt,
COUNT( DISTINCT user_id ) AS shop_user_cnt,
AVG( user_gender_id ) AS shop_user_gender_avg,
AVG( user_age_level ) AS shop_user_age_avg,
AVG( user_occupation_id ) AS shop_user_oppcupation_avg,
AVG( user_star_level ) AS shop_user_star_avg,
COUNT( DISTINCT item_id ) AS shop_item_cnt,
COUNT( DISTINCT item_brand_id ) AS shop_item_brand_cnt,
COUNT( DISTINCT item_city_id ) AS shop_item_city_cnt,
AVG( item_pv_level ) AS shop_item_pv_avg,
AVG( item_price_level ) AS shop_item_price_avg,
AVG( item_sales_level ) AS shop_item_sales_avg,
AVG( item_collected_level ) AS shop_item_collected_avg,
COUNT( DISTINCT context_id ) AS shop_context_cnt,
COUNT( DISTINCT context_page_id ) AS shop_context_page_cnt,
COUNT( DISTINCT context_hour ) AS shop_context_hour_cnt,
AVG( context_hour ) AS shop_context_hour_avg 
FROM
	Summer.data24_raw 
GROUP BY
	shop_id;
	
-- user-item table
SELECT
	user_id,
	item_id,
	COUNT( * ) AS ui_cnt 
FROM
	Summer.data24_raw 
GROUP BY
	user_id,
	item_id 
ORDER BY
	ui_cnt DESC;
	
-- merge tables
SELECT
	* 
FROM
	Summer.data24_raw t1
	LEFT JOIN Summer.user_table t2 ON t1.user_id = t2.user_id
	LEFT JOIN Summer.item_table t3 ON t1.item_id = t3.item_id
	LEFT JOIN Summer.shop_table t4 ON t1.shop_id = t4.shop_id;