# 1. Ortam Hazırlığı ve Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 2. Veri Okuma ve Hazırlama
df = (
    pd.read_excel(r"C:/Users/HARUN TUTAR/Desktop/romanya_elektrik.xlsx", engine="openpyxl")
      .rename(columns={
           "DateTime": "datetime", "Consumption": "consumption_per_hour",
           "Production": "production_per_hour", "Nuclear": "nuclear",
           "Wind": "wind", "Hydroelectric": "hydro",
           "Oil and Gas": "petrol_gas", "Coal": "coal",
           "Solar": "solar", "Biomass": "biomass"
      })
)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").set_index("datetime")
df = df.fillna(method="ffill")

# 3. Özellik Mühendisliği
production_cols = ["nuclear", "wind", "hydro", "petrol_gas", "coal", "solar", "biomass"]
df["total_production"] = df[production_cols].sum(axis=1)
df["net_energy"] = df["total_production"] - df["consumption_per_hour"]
df["hour"] = df.index.hour
df["dow"] = df.index.dayofweek
df["is_weekend"] = df["dow"] >= 5

# 4. Zaman Serisi Özeti
monthly_avg = df.resample("M").mean()
monthly_sum = df.resample("M").sum()
weekly_avg = df.resample("W").mean()

# 5. Görselleştirmeler

# 5.1 Aylık Ortalama Tüketim ve Üretim
plt.figure(figsize=(12,5))
plt.plot(monthly_avg.index, monthly_avg["consumption_per_hour"], label="Aylık Ortalama Tüketim")
plt.plot(monthly_avg.index, monthly_avg["total_production"], label="Aylık Ortalama Üretim")
plt.title("Aylık Ortalama Tüketim ve Üretim")
plt.ylabel("MW"); plt.xlabel("Tarih"); plt.grid(); plt.legend(); plt.show()

# 5.2 Kaynak Bazlı Üretim – Yığılmış Alan Grafiği
monthly_sum[production_cols].plot.area(figsize=(12,5), cmap="tab20")
plt.title("Aylık Üretimde Kaynakların Yığılmış Alan Grafiği")
plt.ylabel("Toplam MW"); plt.xlabel("Tarih"); plt.legend(loc="upper left"); plt.show()

# 5.3 Enerji Karışımı Zamanla Değişimi (%)
mix_pct = monthly_sum[production_cols].div(monthly_sum[production_cols].sum(axis=1), axis=0)
mix_pct.plot(figsize=(12,5))
plt.title("Aylık Enerji Karışımı Zamanla Değişimi (%)")
plt.ylabel("%"); plt.xlabel("Tarih"); plt.legend(loc="upper left"); plt.show()

# 5.4 Aylık Net Enerji Dengesi
plt.figure(figsize=(12,4))
plt.plot(monthly_avg.index, monthly_avg["net_energy"], color="blue", label="Aylık Net Enerji")
plt.axhline(0, color="red", linestyle="--")
plt.title("Aylık Net Enerji (Üretim - Tüketim)")
plt.ylabel("MW"); plt.xlabel("Tarih"); plt.grid(); plt.legend(); plt.show()

# 5.5 Trend/Mevsimsellik/Kalıntı Ayrıştırması
res = seasonal_decompose(monthly_avg["consumption_per_hour"], model="additive")
res.plot().suptitle("Aylık Tüketim Decompose (Trend/Seasonal/Resid)", fontsize=14)
plt.show()

# 6. Haftalık Ortalama Tüketimde Anomali Tespiti
iso = IsolationForest(contamination=0.01, random_state=42)
wf = weekly_avg[["consumption_per_hour"]].copy()
wf["is_anomaly"] = iso.fit_predict(wf[["consumption_per_hour"]]) == -1

plt.figure(figsize=(12,4))
plt.plot(wf.index, wf["consumption_per_hour"], label="Haftalık Ortalama")
plt.scatter(wf[wf["is_anomaly"]].index,
            wf[wf["is_anomaly"]]["consumption_per_hour"],
            color="red", s=80, label="Anomali (IF)")
plt.title("Haftalık Ortalama Tüketimde Anomali Tespiti (Isolation Forest)")
plt.xlabel("Tarih"); plt.ylabel("MW")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# 7. Enerji Kaynakları Arası Korelasyon Matrisi
corr_matrix = df[production_cols].corr(method="pearson")
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Enerji Kaynakları Arasındaki Korelasyon Matrisi")
plt.tight_layout()
plt.show()

# 8. Saatlik Ortalama Tüketim
hourly_avg = df.groupby("hour")["consumption_per_hour"].mean()
colors = ["red" if v > 7000 else "yellow" for v in hourly_avg]
plt.figure(figsize=(10,4))
plt.bar(hourly_avg.index, hourly_avg.values, color=colors)
plt.title("Saatlik Ortalama Tüketim")
plt.xlabel("Saat"); plt.ylabel("MW"); plt.grid(axis="y"); plt.show()

# 9. Üretim vs Tüketim Scatter Plot
plt.figure(figsize=(6,6))
plt.scatter(df["total_production"], df["consumption_per_hour"], alpha=0.3)
m, b = np.polyfit(df["total_production"], df["consumption_per_hour"], 1)
plt.plot(df["total_production"], m*df["total_production"] + b, color="red")
plt.title("Üretim vs Tüketim Scatter & Regresyon")
plt.xlabel("Üretim (MW)"); plt.ylabel("Tüketim (MW)"); plt.grid(); plt.show()

# 10. Trend Bileşeni (Ayrı Görsel)
ts = monthly_avg["consumption_per_hour"]
res = seasonal_decompose(ts, model="additive")
plt.figure(figsize=(10, 4))
plt.plot(res.trend, color="blue", linewidth=2)
plt.title("Aylık Elektrik Tüketimi – Trend Bileşeni")
plt.xlabel("Tarih"); plt.ylabel("MW")
plt.grid(True); plt.tight_layout()
plt.show()

# 11. ACF ve PACF Grafik Analizi
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts, lags=24, zero=False, ax=axes[0])
axes[0].set_title("ACF – Otokorelasyon")
plot_pacf(ts, lags=24, zero=False, method="ywm", ax=axes[1])
axes[1].set_title("PACF – Kısmi Otokorelasyon")
plt.tight_layout()
plt.show()

# 12. Prophet ile 6 Aylık Tüketim Tahmini
df_prop = monthly_avg.reset_index()[["datetime", "consumption_per_hour"]]
df_prop.columns = ["ds", "y"]
m = Prophet()
m.fit(df_prop)
future = m.make_future_dataframe(periods=6, freq="M")
forecast = m.predict(future)

plt.figure(figsize=(10,4))
plt.plot(forecast["ds"], forecast["yhat"], color="blue", label="Tahmin")
plt.plot(forecast[forecast["ds"] > df_prop["ds"].max()]["ds"],
         forecast[forecast["ds"] > df_prop["ds"].max()]["yhat"],
         color="red", linewidth=2.5, label="Tahmin (Gelecek)")
plt.scatter(df_prop["ds"], df_prop["y"], color="gray", s=10, label="Gerçek")
plt.title("Prophet ile 6 Aylık Aylık Tüketim Tahmini")
plt.xlabel("Tarih"); plt.ylabel("MW"); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# 13. Gün ve Saat Bazlı Isı Haritası
pivot = df.pivot_table(index="dow", columns="hour", values="consumption_per_hour", aggfunc="mean")
pivot.index = ["Pt", "Sa", "Ça", "Pe", "Cu", "Ct", "Pz"]
plt.figure(figsize=(12,5))
sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5, linecolor="white")
plt.title("Ortalama Tüketim – Gün ve Saat Bazlı")
plt.ylabel("Gün"); plt.xlabel("Saat")
plt.tight_layout()
plt.show()

# 14. Özet İstatistikler
print("Ortalama Tüketim (MW):", df["consumption_per_hour"].mean().round(2))
print("Ortalama Üretim (MW):", df["total_production"].mean().round(2))
print("Enerji Fazlası Saatler:", (df["net_energy"] > 0).sum())
print("Enerji Açığı Saatler:", (df["net_energy"] < 0).sum())
