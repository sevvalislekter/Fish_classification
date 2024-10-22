# Fish_classification
Bu proje, derin öğrenme tekniklerini kullanarak balık türlerini sınıflandırmayı amaçlamaktadır. Model, görüntü işleme ve yapay sinir ağları (ANN) kullanarak farklı balık türlerini tanımak için eğitilmiştir.

kaggle:https://www.kaggle.com/code/evvalilekter/large-scale-fish-deep-learning

# İlk önce kütühaneleri içeri aktardık.Gerekli kütüphaneler bunlar;

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

Bu proje için kullanılan veri seti, çeşitli balık türlerini içeren görüntülerden oluşmaktadır. Veri seti, eğitim ve doğrulama setleri olarak ikiye ayrılmıştır.


# Veri görselleştirme 
Veri görselleştirme için matplotlib.pyplot kütüphanesi kullanılır.
plt.figure(figsize=(20,10)): Grafiklerin boyutunu ayarlar.
for unique_label in data['label'].unique(): Veri setindeki benzersiz etiketleri döngüye alır.
plt.subplot(3, 3, cn+1): Her bir balık türü için bir alt grafik oluşturur; 3 satır ve 3 sütun şeklinde düzenlenir.
plt.imshow(...): Belirli bir etiket için ilk görüntüyü okur ve görüntüler.
plt.title(unique_label): Görüntünün başlığı olarak balık türünü ekler.
plt.axis('off'): Eksenleri gizler, böylece yalnızca görüntü görünür.

# Veri seti ön işleme 
ImageDataGenerator: ImageDataGenerator, görüntü verilerini artırmak ve ön işlemek için kullanılır. Bu örnekte, görüntülerin piksel değerleri 0-1 aralığına ölçeklenmektedir (rescale=1./255).
flow_from_directory() metodu, belirtilen dizindeki görüntüleri yükler.
 Doğrulama verileri için  bir akış oluşturulur. Eğitim ve doğrulama verileri, validation_split parametresi sayesinde otomatik olarak ayırılır.





# Model eğitimi
Model, Keras ve TensorFlow kullanılarak oluşturulmuştur.
Sequential ile model tanımlayarak başlarız,giriş katmanındaki verileri flatten ile düzleştirir.Gizli katmanlar için relu aktivasyon fonksiyonunu kullanır.Çıkış katmanı için 'softmax' aktivasyon fonksiyonu ile çıkış verir.Optimize edip verilerin yapısı ve katmanı hakkında modal.summary() bilgi verir.

Veri Ön İşleme: Görüntüler 64x64 boyutuna ölçeklendirilmiş ve normalleştirilmiştir.
Model yapısı:
Giriş katmanında görüntüler düzleştirilerek 12288 boyutlu vektöre dönüştürüldü.
Gizli 2 katmanda relu aktivasyon fonksiyonu kullanıldı.
Çıkış katmanında softmax aktivasyon fonksiyonu kullanıldı.

Model eğitiminde 20 epoch eğitim verileri ile eğitildi.
Model.fit() fonksiyonu, modelin eğitim verileri (train_generator) ile eğitilmesini sağlar. validation_data=validation_generator parametresi ile her epoch sonunda modelin performansını değerlendirmek için doğrulama verileri kullanılır.Epochs=20 ile modelin toplamda 20 defa eğitim verileri üzerinde eğitim alması belirtilir. Her epoch, modelin tüm eğitim verisi üzerinden bir geçiş yapmasını ifade eder.

Sonuçları history değişkeninde saklar, bu sayede eğitim ve doğrulama kayıplarını ve doğruluk değerlerini daha sonra analiz edebilirsiniz.


# Sonuç

Modelin performansı, doğrulama verileri üzerinde değerlendirilmiştir. Eğitim kaybı ve doğruluk grafikleri oluşturulmuş, ayrıca karışıklık matrisi ve sınıflandırma raporu ile sonuçlar analiz edilmiştir.
Veri setinin performansını değerlendirir.Model.evaluate(validation_generator) fonksiyonu, modelin doğrulama verileri üzerindeki loss ve accuracy değerlerini hesaplar. Bu, modelin eğitim sırasında öğrendiklerini ne kadar iyi genelleyebildiğini ölçer. test_loss ve test_acc değişkenleri, modelin kayıp ve doğruluk değerlerini saklar.

# tahmin
Tahminlerin Alınması: model.predict(validation_generator) ifadesi, doğrulama veri setindeki görüntüler üzerinde modelin tahminlerini hesaplar. Bu, modelin öğrendiği bilgilere dayanarak her bir görüntü için tahmin edilen sınıf olasılıklarını içerir.

# Karışıklık matrisi ve sınıflandırma raporu
Tahminlerin Alınması: model.predict(validation_generator) ifadesi, modelin doğrulama veri setindeki görüntüler üzerinde yaptığı tahminleri alır. np.argmax(predictions, axis=1) ile tahmin edilen sınıfların indeksleri belirlenir.
true_classes = validation_generator.classes ifadesi, doğrulama veri setindeki gerçek etiketleri alır.

Karışıklık Matrisi: confusion_matrix(true_classes, predicted_classes) ile gerçek ve tahmin edilen sınıflar arasındaki ilişkiyi gösteren karışıklık matrisi oluşturulur. Bu matris, modelin hangi sınıfları doğru tahmin edip etmediğini ve hangi sınıflar arasında karışıklık yaşadığını gösterir.

Sınıflandırma Raporu: classification_report() ile modelin performansı hakkında daha ayrıntılı bilgiler elde edilir. Bu rapor, her bir sınıf için doğruluk, hatırlama (recall), f1 skoru gibi metrikleri içerir.










