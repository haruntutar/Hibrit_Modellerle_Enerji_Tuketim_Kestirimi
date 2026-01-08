# Hibrit Modellerle Enerji Tüketim Analitiği
Bu proje, elektrik tüketim ve üretim verilerini kullanarak enerji arz-talep dengesini analiz eden ve gelecekteki tüketim miktarlarını hibrit makine öğrenmesi modelleri ile tahmin eden kapsamlı bir analitik çalışmadır.

## Proje Özeti
- Türkiye (veya ilgili bölge) enerji şebekesinden alınan saatlik verilerle gerçekleştirilen bu çalışma, iki ana sütun üzerine inşa edilmiştir:
- Keşifsel Veri Analizi (EDA): Kaynak bazlı üretim dağılımı, mevsimsel trendler ve anomali tespiti.
- Tahminleme (Forecasting): İstatistiksel (ARIMA, Holt-Winters) ve modern (Prophet, XGBoost, Random Forest) modellerin yarıştırılarak en yüksek doğruluklu tahminin üretilmesi.

## Öne Çıkan Görselleştirmeler
- Proje kapsamında verinin "nabzını" tutan çeşitli analizler sunulmaktadır:
- Enerji Karışımı Analizi: Nükleer, rüzgar, güneş ve fosil yakıtların üretimdeki payı.
- Tüketim Isı Haritası (Heatmap): Gün ve saat bazlı yoğunluk analizi.
- Anomali Tespiti: Isolation Forest algoritması ile şebekedeki sıra dışı tüketim hareketlerinin belirlenmesi.

## Kullanılan Teknolojiler ve Modeller
- Bu çalışmada, zaman serisi verilerinin karmaşıklığını çözmek için aşağıdaki kütüphane ve modeller kullanılmıştır:

### Veri İşleme: Pandas, NumPy
- Görselleştirme: Matplotlib, Seaborn

### Tahmin Modelleri:
- Prophet: Facebook tarafından geliştirilen, mevsimselliğe dirençli tahminleme. 
- ARIMA & Holt-Winters: Klasik istatistiksel zaman serisi yaklaşımları. 
- XGBoost & Random Forest: Özellik mühendisliği (feature engineering) ile güçlendirilmiş regresyon modelleri. 
- Anomali Tespiti: Scikit-learn IsolationForest 
