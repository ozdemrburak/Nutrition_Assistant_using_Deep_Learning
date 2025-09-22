import streamlit as st
import google.generativeai as genai
from PIL import Image
from get_prediction import predict_image

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="Nutrition Assistant",
    page_icon="🍎",
    layout="wide"
)

# -------------------
# Session State
# -------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None

# -------------------
# Title
# -------------------
st.title("🍎 Beslenme Asistanı Chatbot")
st.markdown("Yiyecek fotoğrafınızı analiz edin ve beslenme hakkında soru sorun!")

# -------------------
# Sidebar - API
# -------------------
with st.sidebar:
    st.header("⚙️ Yapılandırma")
    gemini_api_key = st.text_input(
        "Gemini API Anahtarı",
        type="password",
        help="Google Gemini API anahtarınızı girin"
    )

    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        st.success("✅ API anahtarı yapılandırıldı!")

    st.markdown("---")
    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.chat_history = []
        st.session_state.current_analysis = None
        st.session_state.ai_response = None
        st.rerun()

# -------------------
# Layout: Columns
# -------------------
col1, col2 = st.columns([1, 1])

# -------------------
# COL1: Fotoğraf ve Analiz
# -------------------
with col1:
    st.header("📸 Fotoğraf Yükle")
    uploaded_file = st.file_uploader(
        "Bir yiyecek fotoğrafı seçin...",
        type=['png', 'jpg', 'jpeg'],
        help="Beslenme içeriğini analiz etmek için bir yiyecek fotoğrafı yükleyin"
    )

    if uploaded_file is not None:
        # Görüntüyü göster
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Fotoğraf", width=150)

        if gemini_api_key:
            try:
                with st.spinner("Fotoğraf analiz ediliyor..."):
                    weight, cal, fat, carb, protein = predict_image(uploaded_file).squeeze().tolist()

                # Analiz sonuçlarını session state'e kaydet
                st.session_state.current_analysis = {
                    'weight': weight,
                    'calories': cal,
                    'carbs': carb,
                    'fat': fat,
                    'protein': protein,
                    'image': uploaded_file
                }

                # Ham verileri göster
                st.subheader("🔢 Tespit Edilen Değerler")
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Ağırlık", f"{weight:.1f}g")
                    st.metric("Kalori", f"{cal:.0f} kcal")
                    st.metric("Karbonhidrat", f"{carb:.1f}g")
                with metrics_col2:
                    st.metric("Yağ", f"{fat:.1f}g")
                    st.metric("Protein", f"{protein:.1f}g")

                # İlk analiz promptu ve asistan cevabı
                prediction_key = f"{uploaded_file.name}_{weight:.1f}_{cal:.0f}_{carb:.1f}_{fat:.1f}_{protein:.1f}"
                if 'last_prediction_key' not in st.session_state or st.session_state.get('last_prediction_key') != prediction_key:
                    with st.spinner("İlk analiz yapılıyor..."):
                        try:
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            image_pil = Image.open(uploaded_file)
                            initial_prompt = f"""
                            Bu fotoğraftaki yiyeceği tanımla ve beslenme değerlerini analiz et:

                            Ağırlık: {weight:.1f}g, Kalori: {cal:.0f} kcal, Karbonhidrat: {carb:.1f}g, Yağ: {fat:.1f}g, Protein: {protein:.1f}g

                            Kısa ve öz bir analiz yap (150-200 kelime). Yiyeceği tanımla ve temel beslenme özelliklerini belirt.
                            """
                            response = model.generate_content([initial_prompt, image_pil])

                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': f"📊 **İlk Analiz Tamamlandı!**\n\n{response.text}"
                            })
                            st.session_state.last_prediction_key = prediction_key

                        except Exception as e:
                            st.error(f"Analiz hatası: {str(e)}")

            except Exception as e:
                st.error(f"Fotoğraf işlenirken hata: {str(e)}")

# -------------------
# COL2: Scrollable Chat (Otomatik Scroll)
# -------------------
with col2:
    st.header("💬 Beslenme Sohbeti")

    chat_container = st.empty()  # Boş konteyner

    # Kullanıcıdan input al
    if st.session_state.current_analysis and gemini_api_key:
        user_question = st.chat_input("Beslenme hakkında soru sorun...")

        if user_question:
            # Kullanıcı mesajını kaydet
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })

            # Asistan cevabı
            try:
                with st.spinner("Cevap hazırlanıyor..."):
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    analysis = st.session_state.current_analysis

                    context = f"""
                    Kullanıcının yüklediği yiyecek hakkında şu veriler var:
                    Ağırlık: {analysis['weight']:.1f}g
                    Kalori: {analysis['calories']:.0f} kcal
                    Karbonhidrat: {analysis['carbs']:.1f}g
                    Yağ: {analysis['fat']:.1f}g
                    Protein: {analysis['protein']:.1f}g

                    Sohbet geçmişi:
                    {chr(10).join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])}

                    Kullanıcı sorusu: {user_question}

                    Sadece bu beslenme verilerine dayanarak cevap ver. Kısa ve anlaşılır ol (100-200 kelime). Türkçe cevapla.
                    """

                    image_pil = Image.open(analysis['image'])
                    response = model.generate_content([context, image_pil])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.text
                    })

            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Üzgünüm, bir hata oluştu: {str(e)}"
                })

    # Mesajları container içinde göster (otomatik scroll için)
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

# -------------------
# Nasıl Kullanılır
# -------------------
with st.expander("📋 Nasıl Kullanılır"):
    st.markdown("""
    **Adım adım kullanım:**

    1. **API Anahtarı**: Kenar çubuğuna Gemini API anahtarınızı girin.
    2. **Fotoğraf Yükle**: Sol taraftan bir yiyecek fotoğrafı seçin.
    3. **İlk Analiz**: Sistem otomatik olarak beslenme analizini yapar.
    4. **Soru Sor**: Sağ taraftaki chatbot'a istediğiniz soruyu sorun.

    **API Anahtarı almak için:**
    - [Google AI Studio](https://aistudio.google.com/app/apikey) adresine gidin.
    - Yeni bir API anahtarı oluşturun.
    - Eğer API key oluşturamazsanız iletişime geçiniz: ozdemrburak@yahoo.com
    """)
