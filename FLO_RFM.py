
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################
           # 1. flo_data_20K.csv verisini okuyunuz.
           # 2. Veri setinde
                     # a. İlk 10 gözlem,
                     # b. Değişken isimleri,
                     # c. Betimsel istatistik,
                     # d. Boş değer,
                     # e. Değişken tipleri, incelemesi yapınız.
           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
           # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# flo_data_20K.csv verisini okuyunuz
df_ = pd.read_csv("FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy()

# İlk 10 gözlem
df.head(10)

# Değişken isimleri
df.columns

# Betimsel istatistik
df.describe().T

# Boş değer
df.isnull().sum()

# Değişken tipleri ve Genel Bilgi
df.dtypes
df.info()

# Herbir müşterinin toplam alışveriş sayısı
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# Herbir müşterinin harcaması
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Tarih ifade eden değişkenler: first_order_date, last_order_date, last_order_date_online, last_order_date_offline
for col in df.columns:
    if "date" in col:
        df[col]= df[col].apply(pd.to_datetime)

# Bir başka yol
contains_date =  df.columns[df.columns.str.contains("date")]
df[contains_date] = df[contains_date].apply(pd.to_datetime)

df.dtypes

# Alışveriş kanallarındaki müşteri sayısının,
# toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({"master_id": ["count"],
                                "order_num_total": ["sum"],
                                "customer_value_total": ["sum"]})

# En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"customer_value_total": "sum"}).sort_values("customer_value_total", ascending=False).head(10)

# En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"order_num_total": "sum"}).sort_values("order_num_total", ascending=False).head(10)


#Veri ön hazırlık sürecini fonksiyonlaştırınız
def data_prep(dataframe):
    # Herbir müşterinin toplam alışveriş sayısı
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]

    # Herbir müşterinin harcaması
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    # Tarih ifade edip object olan değişkenlerin tipini date'e çevirelim
    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = dataframe[col].apply(pd.to_datetime)

    return dataframe

df = df_.copy()
data_prep(df)




###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

# Recency, Frequency ve Monetary
df.head()
df["last_order_date"].max()

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrası analiz tarihi olarak kabul edilebilir.
today_date = dt.datetime(2021, 6, 1)

type(today_date)

#customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe
rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'order_num_total': lambda order_num_total: order_num_total,
                                     'customer_value_total': lambda customer_value_total: customer_value_total})
rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T
rfm.shape




###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi
# Bu skorların recency_score, frequency_score ve monetary_score olarak kaydedilmesi
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

#rank(method="first") // İlk gördüğünü ilk sınıfa ata
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

rfm[rfm["RF_SCORE"] == "54"]




###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama
# ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme

# RFM isimlendirmesi
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
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])



# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde.
# Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçebilmek isteniliyor.
# Sadık müşterilerinden(champions,loyal_customers), ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.


flo_kadın= rfm.loc[(rfm["segment"] == "loyal_customers") | (rfm["segment"] == "champions")
                         & rfm["interested_in_categories_12"].str.contains("KADIN")]
flo_kadın.info()


# Sor?
target_segments = rfm[rfm["segment"].isin(["champions", "loyal_customers"])].index
target_cust = df[(df["master_id"].isin(target_segments)) &
                 (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]

target_cust.to_csv("yeni_marka_hedef_musteri_id.cvs")
target_cust.shape


#bir baska yol:
new_df = pd.DataFrame({'segment': rfm['segment'], 'interested_in_categories_12': df['interested_in_categories_12']})
new_df.head()

flo_kadin = new_df.loc[(new_df["segment"].isin(["champions", 'loyal_customers'])) &
           (new_df["interested_in_categories_12"].str.contains("KADIN"))]





#new_df = pd.DataFrame()
#new_df["new_customer_id"] = df[rfm.index.isin(rfm[(rfm["segment"] == "champions")
#                                                  | (rfm["segment"] == "loyal_customers")].index)
#                               & ((df["interested_in_categories_12"].str.contains("KADIN")))]["master_id"]

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır.
# Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
# alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor.
# Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv olarak kaydediniz.

new_target_segments = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])].index
new_target_cust = df[(df["master_id"].isin(new_target_segments)) &
               ((df["interested_in_categories_12"].str.contains("ERKEK"))|
                (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]

new_target_cust.to_csv("indirim_hedef_müşteri_ids.csv")

new_target_cust.shape

#yada
flo_40 = new_df.loc[(new_df["segment"].isin(["cant_loose", "about_to_sleep", 'new_customers'])) &
           (new_df["interested_in_categories_12"].str.contains("COCUK | ERKEK"))]

