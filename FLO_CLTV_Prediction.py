##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma Model
##############################################################
###############################################################
# Task 1: Preparing data
###############################################################
import datetime as dt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


df_ = pd.read_csv("FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()
df.head()
df.info()
df.isnull().sum()
df.describe([0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

#Define the outlier_thresholds and replace_with_thresholds functions required to suppress outliers
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
    
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


# Determine if variables "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline" and "customer_value_total_ever_online" have contradictory values

for col in df.columns:
    if df[col].dtypes != "O":
        replace_with_thresholds(df, col)

df.describe([0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T



# Omnichannel states that customers shop both online and offline platforms. Create new variables for
# the total number of shopping and spending of each customer

df["total_number_of_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]




# Examine the variable types. Turn the type of variables containing "date" to the datetime type.

contains_date = df.columns[df.columns.str.contains("date")]
df[contains_date] = df[contains_date].apply(pd.to_datetime)


###############################################################
# TASK 2: Creating the CLTV Data Structure
###############################################################

df["last_order_date"].max()
analysis_date = dt.datetime(2021, 6, 1)


# Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, Frequency and Monetary_CLTV_AVG values.

# recency: The amount of time that has passed since the customer's last purchase. Measured on a weekly basis. (calculated for each individual user)
# T: Customer tenure. Measured on a weekly basis. It represents the time that has elapsed since the customer's first purchase from the analysis date.
# frequency: The total number of repeat purchases made by the customer. This metric considers only purchases with a frequency greater than 1.
# monetary: The average earnings or revenue generated per purchase. It indicates the average amount of money spent by the customer in each transaction.


cltv_df = pd.DataFrame({"customer_id": df["master_id"],
                        "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
                        "T_weekly": ((analysis_date - df["first_order_date"]).dt.days)/7,
                        "frequency": df["total_number_of_purchases"],
                        "monetary_cltv_avg": df["total_price"] / df["total_number_of_purchases"]})




###############################################################
# TASK 3: Calculation of CLTV using BG/NBD and Gamma-Gamma
###############################################################
# 1. Fit the BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# a. Estimate the expected purchases from customers in 3 months and add to the CLTV DataFrame as exp_sales_3_month.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df["frequency"],
                                       cltv_df["recency_cltv_weekly"],
                                       cltv_df["T_weekly"])


# b. Estimate the expected purchases from customers in 6 months and add to the CLTV DataFrame as exp_sales_6_month.

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df["frequency"],
                                       cltv_df["recency_cltv_weekly"],
                                       cltv_df["T_weekly"])

cltv_df.head(10)



# 2. Fit the Gamma-Gamma model. Estimate the average value of the customers and 
#add it to the cltv dataframe as exp_average_value.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])
cltv_df.head(10)



# 3. Calculate 6 months CLTV and add it to the dataframe with the name cltv.
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6,    # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)


# b. Observe 20 people at the highest value of CLTV.
cltv_df.sort_values("cltv", ascending=False).head(20)




###############################################################
# TASK 4: Creating Segments by CLTV
###############################################################
# Divide all your customers into 4 groups (segments)

cltv_df["SEGMENT"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.groupby("SEGMENT").agg({"count", "sum", "mean"})



