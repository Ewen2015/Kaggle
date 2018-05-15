-- Project: Conversion Rate Prediction
-- Date: May 6, 2018
-- Author: Zilin, Ewen

-- ================================
-- preprocess
-- update tables
update round2_train
set
context_realtime = date_format(from_unixtime(context_timestamp), '%y-%m-%d %h:%i:%s'),
context_day = extract(day from context_realtime),
context_hour = extract(hour from context_realtime),
shopping_hour = case 
       when context_hour < 4 then context_hour + 20
       else context_hour - 4
       end,
num_category = 1 + length(item_category_list) - length(replace(item_category_list, ';', '')),
item_category_1 = substring_index(substring_index(item_category_list, ';', 2), ';', -1),
item_category_2 = case 
       when num_category=3 then substring_index(substring_index(item_category_list, ';', 3), ';', -1)
       else null 
       end;

update round2_test_a
set
context_realtime = date_format(from_unixtime(context_timestamp), '%y-%m-%d %h:%i:%s'),
context_day = extract(day from context_realtime),
context_hour = extract(hour from context_realtime),
shopping_hour = case 
       when context_hour < 4 then context_hour + 20
       else context_hour - 4
       end,
num_category = 1 + length(item_category_list) - length(replace(item_category_list, ';', '')),
item_category_1 = substring_index(substring_index(item_category_list, ';', 2), ';', -1),
item_category_2 = case 
       when num_category=3 then substring_index(substring_index(item_category_list, ';', 3), ';', -1)
       else null 
       end;
       
-- update round2_test_b
-- set
-- context_realtime = date_format(from_unixtime(context_timestamp), '%y-%m-%d %h:%i:%s'),
-- context_day = extract(day from context_realtime),
-- context_hour = extract(hour from context_realtime),
-- shopping_hour = case 
--     when context_hour < 4 then context_hour + 20
--     else context_hour - 4
--     end;
-- num_category = 1 + length(item_category_list) - length(replace(item_category_list, ';', '')),
-- item_category_1 = substring_index(substring_index(item_category_list, ';', 2), ';', -1),
-- item_category_2 = case 
--     when num_category=3 then substring_index(substring_index(item_category_list, ';', 3), ';', -1)
--     else null 
--     end;

-- ================================
-- seperate training data by day 7
-- create tables

create table summer.round2_train_7 as
select * 
from summer.round2_train
where context_day = 7;

create table summer.round2_train_past as
select * 
from summer.round2_train
where not context_day = 7;

-- ================================
-- feature engineering
-- mat behaivor model 

-- user table
create table mat_user_table_day_7 as 
select
user_id,
count( * ) as user_instance_cnt,
count( distinct item_id ) as user_item_cnt,
count( * ) / count( distinct item_id ) as user_item_avg,
avg( item_price_level ) as user_price_avg,
stddev( item_price_level ) as user_price_sd,
max( item_price_level ) as user_price_max,
min( item_price_level ) as user_price_min,
avg( item_pv_level) as user_item_pv_avg,
stddev( item_pv_level) as user_item_pv_sd,
max( item_pv_level) as user_item_pv_max,
min( item_pv_level) as user_item_pv_min,
avg( item_price_level) as user_item_price_avg,
stddev( item_price_level) as user_item_price_sd,
max( item_price_level) as user_item_price_max,
min( item_price_level) as user_item_price_min,
avg( item_sales_level ) as user_item_sales_avg,
stddev( item_sales_level ) as user_item_sales_sd,
max( item_sales_level ) as user_item_sales_max,
min( item_sales_level ) as user_item_sales_min,
avg( item_collected_level ) as user_item_collected_avg,
stddev( item_collected_level ) as user_item_collected_sd,
max( item_collected_level ) as user_item_collected_max,
min( item_collected_level ) as user_item_collected_min,
count( distinct context_id ) as user_context_cnt,
count( distinct context_page_id ) as user_context_page_cnt,
count( distinct shop_id ) as user_shop_cnt,
avg( shop_review_num_level ) as user_shop_review_num_avg,
stddev( shop_review_num_level ) as user_shop_review_num_sd,
max( shop_review_num_level ) as user_shop_review_num_max,
min( shop_review_num_level ) as user_shop_review_num_min,
avg( shop_review_positive_rate ) as user_shop_review_positive_rate_avg,
stddev( shop_review_positive_rate ) as user_shop_review_positive_rate_sd,
max( shop_review_positive_rate ) as user_shop_review_positive_rate_max,
min( shop_review_positive_rate ) as user_shop_review_positive_rate_min,
avg( shop_star_level ) as user_shop_star_avg,
stddev( shop_star_level ) as user_shop_star_sd,
avg( shop_score_service ) as user_shop_score_service_avg,
stddev( shop_score_service ) as user_shop_score_service_sd,
max( shop_score_service ) as user_shop_score_service_max,
min( shop_score_service ) as user_shop_score_service_min,
avg( shop_score_delivery ) as user_shop_score_delivery_avg,
stddev( shop_score_delivery ) as user_shop_score_delivery_sd,
max( shop_score_delivery ) as user_shop_score_delivery_max,
min( shop_score_delivery ) as user_shop_score_delivery_min,
avg( shop_score_description ) as user_shop_score_description_avg,
stddev( shop_score_description ) as user_shop_score_description_sd,
max( shop_score_description ) as user_shop_score_description_max,
min( shop_score_description ) as user_shop_score_description_min 
from
       round2_train_7
group by
       user_id;
       
-- item table
create table mat_item_table_day_7 as 
select
item_id,
count( * ) as item_instance_cnt,
count( distinct user_id ) as item_user_cnt,
count( * ) / count( distinct user_id ) as item_user_avg,
avg( user_gender_id ) as item_user_gender_avg,
avg( user_age_level ) as item_user_age_avg,
avg( user_occupation_id ) as item_user_oppcupation_avg,
avg( user_star_level ) as item_user_star_avg,
count( is_trade = 1 ) as item_is_trade_count
from
       round2_train_7
group by
       item_id;

-- item table before day 7
create table mat_item_table_past as 
select
item_id,
count( * ) as item_instance_past_cnt,
count( distinct user_id ) as item_user_past_cnt,
count( * ) / count( distinct user_id ) as item_user_past_avg,
count( is_trade = 1 ) as item_is_trade_past_count,
count( is_trade = 1 ) / count( * ) as item_past_cvr,
avg( user_gender_id ) as item_user_gender_past_avg,
avg( user_age_level ) as item_user_age_past_avg,
avg( user_occupation_id ) as item_user_oppcupation_past_avg,
avg( user_star_level ) as item_user_star_past_avg
from
       round2_train_past
group by
       item_id;
       
-- user table before day 7
create table mat_user_table_past as 
select
user_id,
count( * ) as user_instance_past_cnt,
count( distinct item_id ) as user_item_past_cnt,
count( * ) / count( distinct item_id ) as user_item_past_avg,
avg( shopping_hour ) as user_shopping_hour_avg,
stddev( shopping_hour ) as user_shopping_hour_sd,
max( shopping_hour ) as user_shopping_hour_max,
min( shopping_hour ) as user_shopping_hour_min,
avg( item_price_level ) as user_price_past_avg,
stddev( item_price_level ) as user_price_past_sd,
max( item_price_level ) as user_price_past_max,
min( item_price_level ) as user_price_past_min,
avg( item_pv_level) as user_item_pv_past_avg,
stddev( item_pv_level) as user_item_pv_past_sd,
max( item_pv_level) as user_item_pv_past_max,
min( item_pv_level) as user_item_pv_past_min,
avg( item_price_level) as user_item_price_past_avg,
stddev( item_price_level) as user_item_price_past_sd,
max( item_price_level) as user_item_price_past_max,
min( item_price_level) as user_item_price_past_min,
avg( item_sales_level ) as user_item_sales_past_avg,
stddev( item_sales_level ) as user_item_sales_past_sd,
max( item_sales_level ) as user_item_sales_past_max,
min( item_sales_level ) as user_item_sales_past_min,
avg( item_collected_level ) as user_item_collected_past_avg,
stddev( item_collected_level ) as user_item_collected_past_sd,
max( item_collected_level ) as user_item_collected_past_max,
min( item_collected_level ) as user_item_collected_past_min
from
       round2_train_past
group by
       user_id;

-- ================================
-- merge tables

create table mat_test_a as
select * 
from summer.round2_test_a 
left join summer.mat_user_table_day_7 using (user_id)
left join summer.mat_item_table_day_7 using (item_id)
left join summer.mat_user_table_past using (user_id)
left join summer.mat_item_table_past using (item_id)
;

create table mat_train_day7 as
select * 
from summer.round2_test_a 
left join summer.mat_user_table_day_7 using (user_id)
left join summer.mat_item_table_day_7 using (item_id)
left join summer.mat_user_table_past using (user_id)
left join summer.mat_item_table_past using (item_id)
;

-- create table mat_test_b as
-- select t1.* 
-- from summer.round2_test_a t1
-- left join summer.mat_user_table_day_7 t2 on t1.user_id = t2.user_id
-- left join summer.mat_item_table_day_7 t3 on t1.item_id = t3.item_id
-- left join summer.mat_user_table_past t4 on t1.user_id = t4.user_id
-- left join summer.mat_item_table_past t5 on t1.item_id = t5.item_id
-- ;

