create database bigdproj;

use bigdproj;

drop table tweets;

create table tweets (obs int, text string, created_at timestamp, geo string, lang string, place string, coordinates string, user_favourites_count int, user_statuses_count int, user_description string, user_location string, user_id float, user_created_at timestamp, user_verified boolean, user_following boolean, user_url string, user_listed_count int, user_followers_count int, user_default_profile_image boolean, user_utc_offset int, user_friends_count int, user_default_profile boolean, user_name string, user_lang string, user_screen_name string, user_geo_enabled boolean, user_profile_background_color string, user_profile_image_url string, user_time_zone string, id float, favorite_count int, retweeted boolean, source string, favorited boolean, retweet_count int) row format serde 'org.apache.hadoop.hive.serde2.OpenCSVSerde' with serdeproperties ("escapeChar" = "Æ") tblproperties ("skip.header.line.count"="1");

load data local inpath '/root/project/election_day.csv' overwrite into table tweets;

select count(text) from tweets;

create table twtnew (text string, created_at timestamp, favorite_count int, retweet_count int) row format serde 'org.apache.hadoop.hive.serde2.OpenCSVSerde' with serdeproperties ("escapeChar" = "Æ") tblproperties("skip.header.line.count"="1");

insert overwrite table twtnew select text, created_at, favorite_count, retweet_count from tweets where (rand() <= 0.1 and lang='en') distribute by rand() sort by rand();

select count(text) from twtnew;

alter table twtnew add columns (is_retweeted boolean, rem_time bigint);

select max(cast(cast(created_at as timestamp) as int)) from twtnew;


insert overwrite table twtnew select text, created_at, favorite_count, retweet_count, if(retweet_count>0, TRUE, FALSE) as is_retweeted, 1478696895-cast(cast(created_at as timestamp) as int) as rem_time from twtnew;

select is_retweeted, count(text), sum(retweet_count), sum(favorite_count) from twtnew where (lower(text) rlike '.*hillary.*' or lower(text) rlike '.*clinton.*') group by is_retweeted;


select is_retweeted, count(text), sum(retweet_count), sum(favorite_count) from twtnew where (lower(text) rlike '.*donald.*' or lower(text) rlike '.*trump.*') group by is_retweeted;


select min(cast(rem_time as int)) from twtnew;

select max(cast(rem_time as int)) from twtnew;


create table textcount as
select tab.rem_time_bin, count(tab.twt) as txtcnt
from
(select text as twt, 
case 
when cast(rem_time as int)<=12000 then '10th Period'
when cast(rem_time as int)>12000 and cast(rem_time as int)<=24000 then '09th Period'
when cast(rem_time as int)>24000 and cast(rem_time as int)<=36000 then '08th Period'
when cast(rem_time as int)>36000 and cast(rem_time as int)<=48000 then '07th Period'
when cast(rem_time as int)>48000 and cast(rem_time as int)<=60000 then '06th Period'
when cast(rem_time as int)>60000 and cast(rem_time as int)<=72000 then '05th Period'
when cast(rem_time as int)>72000 and cast(rem_time as int)<=84000 then '04th Period'
when cast(rem_time as int)>84000 and cast(rem_time as int)<=96000 then '03rd Period'
when cast(rem_time as int)>96000 and cast(rem_time as int)<=108000 then '02nd Period'
when cast(rem_time as int)>108000 and cast(rem_time as int)<=120000 then '01st Period'
end
as rem_time_bin
from twtnew) as tab
group by tab.rem_time_bin;


select * from textcount;


create table hilcount as 
select tab.rem_time_bin, count(tab.twt) as hilcnt
from
(select text as twt, 
case 
when cast(rem_time as int)<=12000 then '10th Period'
when cast(rem_time as int)>12000 and cast(rem_time as int)<=24000 then '09th Period'
when cast(rem_time as int)>24000 and cast(rem_time as int)<=36000 then '08th Period'
when cast(rem_time as int)>36000 and cast(rem_time as int)<=48000 then '07th Period'
when cast(rem_time as int)>48000 and cast(rem_time as int)<=60000 then '06th Period'
when cast(rem_time as int)>60000 and cast(rem_time as int)<=72000 then '05th Period'
when cast(rem_time as int)>72000 and cast(rem_time as int)<=84000 then '04th Period'
when cast(rem_time as int)>84000 and cast(rem_time as int)<=96000 then '03rd Period'
when cast(rem_time as int)>96000 and cast(rem_time as int)<=108000 then '02nd Period'
when cast(rem_time as int)>108000 and cast(rem_time as int)<=120000 then '01st Period'
end
as rem_time_bin
from twtnew
where (lower(text) rlike '.*hillary.*' or lower(text) rlike '.*clinton.*')) as tab
group by tab.rem_time_bin;


select * from hilcount;



create table trpcount as 
select tab.rem_time_bin, count(tab.twt) as trpcnt
from
(select text as twt, 
case 
when cast(rem_time as int)<=12000 then '10th Period'
when cast(rem_time as int)>12000 and cast(rem_time as int)<=24000 then '09th Period'
when cast(rem_time as int)>24000 and cast(rem_time as int)<=36000 then '08th Period'
when cast(rem_time as int)>36000 and cast(rem_time as int)<=48000 then '07th Period'
when cast(rem_time as int)>48000 and cast(rem_time as int)<=60000 then '06th Period'
when cast(rem_time as int)>60000 and cast(rem_time as int)<=72000 then '05th Period'
when cast(rem_time as int)>72000 and cast(rem_time as int)<=84000 then '04th Period'
when cast(rem_time as int)>84000 and cast(rem_time as int)<=96000 then '03rd Period'
when cast(rem_time as int)>96000 and cast(rem_time as int)<=108000 then '02nd Period'
when cast(rem_time as int)>108000 and cast(rem_time as int)<=120000 then '01st Period'
end
as rem_time_bin
from twtnew
where (lower(text) rlike '.*donald.*' or lower(text) rlike '.*trump.*')) as tab
group by tab.rem_time_bin;


select * from trpcount;



create table hilpercent as
select hilcount.rem_time_bin as rem_time_bin, hilcount.hilcnt/textcount.txtcnt as hilpct
from hilcount join textcount
on hilcount.rem_time_bin = textcount.rem_time_bin;

select * from hilpercent;


create table trppercent as
select trpcount.rem_time_bin as rem_time_bin, trpcount.trpcnt/textcount.txtcnt as trppct
from trpcount join textcount
on trpcount.rem_time_bin = textcount.rem_time_bin;


select * from trppercent;


select corr(hilpercent.hilpct, trppercent.trppct) 
from hilpercent join trppercent
on hilpercent.rem_time_bin = trppercent.rem_time_bin;


insert overwrite local directory '/root/project/results/trppercent.csv' 
row format delimited fields terminated by ','
select * from trppercent;

insert overwrite local directory '/root/project/results/hilpercent' 
row format delimited fields terminated by ','
select * from hilpercent;



