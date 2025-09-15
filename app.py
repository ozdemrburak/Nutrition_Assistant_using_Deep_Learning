import streamlit as st
from get_prediction import predict_image

st.title("Yemek Besin Değerleri Tahmini")

uploaded_file = st.file_uploader("Bir yemek fotoğrafı yükle", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Yüklenen Görsel", use_column_width=True)
    weight, cal, carb, fat, protein = predict_image(uploaded_file).squeeze().tolist()

    st.subheader("📊 Tahmini Besin Değerleri")
    st.write(f"Yemeğin Gramajı: {weight:.1f} gram")
    st.write(f"Kalori: {cal:.1f} kcal")
    st.write(f"Karbonhidrat: {carb:.1f} g")
    st.write(f"Yağ: {fat:.1f} g")
    st.write(f"Protein: {protein:.1f} g")
