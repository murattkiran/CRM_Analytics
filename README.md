**RFM ANALYSIS FOR CUSTOMER SEGMENTATION AND CLTV PREDICTION**

This project aims to segment customers of FLO, one of the largest shoe companies in Turkey, and determine marketing strategies based on these segments. The project will utilize RFM analysis for customer segmentation and BG-NBD and Gamma-Gamma models for Customer Lifetime Value (CLTV) prediction.

The dataset used for this project consists of historical shopping behavior of customers who made purchases through OmniChannel (both online and offline) in the years 2020-2021. The dataset includes the following important columns:

    master_id: Unique customer number
    order_channel: Platform from which the purchase was made (Android, iOS, Desktop, Mobile, Offline)
    last_order_channel: Channel from which the last purchase was made
    first_order_date: Date of the first purchase made by the customer
    last_order_date: Date of the last purchase made by the customer
    last_order_date_online: Date of the last online purchase made by the customer
    last_order_date_offline: Date of the last offline purchase made by the customer
    order_num_total_ever_online: Total number of purchases made by the customer online
    order_num_total_ever_offline: Total number of purchases made by the customer offline
    customer_value_total_ever_offline: Total amount spent by the customer in offline purchases
    customer_value_total_ever_online: Total amount spent by the customer in online purchases
    interested_in_categories_12: List of categories in which the customer made purchases in the last 12 months

**Business Problem**

FLO aims to segment its customers and determine marketing strategies based on these segments. The goal of this project is to define customer behaviors and create groups based on these behavioral patterns. Customized offers will be provided to customers based on these segments, and Customer Lifetime Value (CLTV) predictions will be made.
