
###############################################################
# Customer Segmentation with RFM
###############################################################

###############################################################
# TASK 1: Data Understanding
###############################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df_ = pd.read_csv("FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy()

# The first 10 observations
df.head(10)

# Variable names
df.columns

# Descriptive statistics
df.describe().T

# Null value
df.isnull().sum()

# Variable types and general information
df.dtypes
df.info()

# Total number of purchases for each customer
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# Total spending for each customer
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Convert the object variables containing date in the data set to date format.
for col in df.columns:
    if "date" in col:
        df[col]= df[col].apply(pd.to_datetime)

# or
contains_date =  df.columns[df.columns.str.contains("date")]
df[contains_date] = df[contains_date].apply(pd.to_datetime)

df.dtypes

# Look at the distribution of the number of customers in the shopping channels, the total number of products
# purchased and total expenditures.
df.groupby("order_channel").agg({"master_id": ["count"],
                                "order_num_total": ["sum"],
                                "customer_value_total": ["sum"]})

# Sort the top 10 customers with the highest total revenue.
df.groupby("master_id").agg({"customer_value_total": "sum"}).sort_values("customer_value_total", ascending=False).head(10)

# Sort the top 10 customers with the highest number of orders.
df.groupby("master_id").agg({"order_num_total": "sum"}).sort_values("order_num_total", ascending=False).head(10)


# Functionalize the data preparation process.
def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    contains_date =  df.columns[df.columns.str.contains("date")]
    df[contains_date] = df[contains_date].apply(pd.to_datetime)
    return dataframe

df = df_.copy()
data_prep(df)




###############################################################
# TASK 2: Calculating RFM Metrics
###############################################################

# Recency, Frequency ve Monetary
df.head()
df["last_order_date"].max()

# Setting the recency date for 2 days after the last order date
today_date = dt.datetime(2021, 6, 1)

type(today_date)

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'order_num_total': lambda order_num_total: order_num_total,
                                     'customer_value_total': lambda customer_value_total: customer_value_total})
rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T
rfm.shape




###############################################################
# TASK 3: Calculating RF and RFM Scores
###############################################################

# Converting Recency, Frequency and Monetary metrics to scores between 1-5 with the help of qcut and recording
# these scores as recency_score, frequency_score and monetary_score

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

rfm[rfm["RF_SCORE"] == "54"]




###############################################################
#TASK 4: Defining RF Scores as Segments
###############################################################

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)


###############################################################
###############################################################

# 1. Examine the recency, frequency and monetary averages of the segments.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# 2. With the help of RFM analysis, find the customers in the relevant profile for the 2 cases given below and save the customer ids as csv
# a. FLO is incorporating a new women's footwear brand. The product prices of the brand it includes are determined by the general customer
# above their preferences. Therefore, for the promotion of the brand and product sales, we specially contact customers with the profile of interest
# to get in touch with you. Loyal customers (champions, loyal_customers) and shopping in the female category
# are customers to be specially contacted. Save the id numbers of these customers in the csv file.


target_segments = rfm[rfm["segment"].isin(["champions", "loyal_customers"])].index
target_cust = df[(df["master_id"].isin(target_segments)) &
                 (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]

target_cust.to_csv("yeni_marka_hedef_musteri_id.cvs")
target_cust.shape



# b. Up to 40% discount is planned for Men's and Children's products. We want to specifically target customers who
# are good customers in the past who are interested in categories related to this discount, but have not shopped for
# a long time and new customers. Save the ids of the customers in the appropriate profile to the csv file as
# discount_target_customer_ids.csv.

new_target_segments = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])].index
new_target_cust = df[(df["master_id"].isin(new_target_segments)) &
               ((df["interested_in_categories_12"].str.contains("ERKEK"))|
                (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]

new_target_cust.to_csv("indirim_hedef_müşteri_ids.csv")

new_target_cust.shape


