import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# 2. Streamlit uygulamanızın arayüzünü oluşturun
st.title("Yorum Duygu Analizi")

# 3. Kullanıcıdan bir yorum girmesini isteyen bir metin giriş alanı ekleyin
user_input = st.text_area("Lütfen bir yorum yazın ve ardından Ctrl + Enter tuşlarına basın:")

# 4. Kullanıcının girdiği yorumu analiz etmek için modelinizi kullanın
st.cache_data
def load_model():
    # Eğittiğiniz modeli yükleyin (USE4.model olarak kaydedilmiş gibi görünüyor)
    model = tf.keras.models.load_model("USE4.model")
    return model

model = load_model()

if user_input:
    # Kullanıcının girdiği yorumu modelin beklediği formata dönüştürün
    user_input = [user_input]

    # 5. Modeli kullanarak yorumu analiz edin
    pred_prob = model.predict(user_input)
    pred_label = int(np.round(pred_prob))
    prediction = "Olumlu" if pred_label == 1 else "Olumsuz"
    confidence = pred_prob[0][0]

    # Sonucu kullanıcıya gösterin
    st.write(f"Analiz Sonucu: Bu yorum {prediction} ({confidence:.2f} güven düzeyi)")

# Streamlit uygulamanınızı çalıştırmak için terminalde aşağıdaki komutu kullanın:
# streamlit run duygu_analizi.py
