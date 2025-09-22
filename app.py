import streamlit as st
import google.generativeai as genai
from PIL import Image
from get_prediction import predict_image

# Configure page
st.set_page_config(
    page_title="Nutrition Assistant",
    page_icon="🍎",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None

# Title and description
st.title("🍎 Beslenme Asistanı Chatbot")
st.markdown("Yiyecek fotoğrafınızı analiz edin ve beslenme hakkında soru sorun!")

# Sidebar for API configuration
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

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Fotoğraf Yükle")
    uploaded_file = st.file_uploader(
        "Bir yiyecek fotoğrafı seçin...",
        type=['png', 'jpg', 'jpeg'],
        help="Beslenme içeriğini analiz etmek için bir yiyecek fotoğrafı yükleyin"
    )

    if uploaded_file is not None:
        # Display uploaded image (küçültülmüş boyut)
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Fotoğraf", width=200)

        if gemini_api_key:
            try:
                with st.spinner("Fotoğraf analiz ediliyor..."):
                    # Process image with SigLIP2 regressor
                    weight, cal, fat, carb, protein = predict_image(uploaded_file).squeeze().tolist()

                # Store analysis results
                st.session_state.current_analysis = {
                    'weight': weight,
                    'calories': cal,
                    'carbs': carb,
                    'fat': fat,
                    'protein': protein,
                    'image': uploaded_file
                }

                # Display raw predictions
                st.subheader("🔢 Tespit Edilen Değerler")
                metrics_col1, metrics_col2 = st.columns(2)

                with metrics_col1:
                    st.metric("Ağırlık", f"{weight:.1f}g")
                    st.metric("Kalori", f"{cal:.0f} kcal")
                    st.metric("Karbonhidrat", f"{carb:.1f}g")

                with metrics_col2:
                    st.metric("Yağ", f"{fat:.1f}g")
                    st.metric("Protein", f"{protein:.1f}g")

                # Initial analysis if not done yet
                prediction_key = f"{uploaded_file.name}_{weight:.1f}_{cal:.0f}_{carb:.1f}_{fat:.1f}_{protein:.1f}"

                if 'last_prediction_key' not in st.session_state or st.session_state.get(
                        'last_prediction_key') != prediction_key:
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

                            # Add to chat history
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': f"📊 **İlk Analiz Tamamlandı!**\n\n{response.text}"
                            })
                            st.session_state.last_prediction_key = prediction_key

                        except Exception as e:
                            st.error(f"Analiz hatası: {str(e)}")

            except Exception as e:
                st.error(f"Fotoğraf işlenirken hata: {str(e)}")

with col2:
    st.header("💬 Beslenme Sohbeti")

    # Display chat history in scrollable container (HTML kodu gözükme sorunu fix)
    if st.session_state.chat_history:
        chat_html = "<div id='chat-box' style='height:400px; overflow-y:auto; border:1px solid #e0e0e0; border-radius:10px; padding:15px; background-color:#fafafa; margin-bottom:20px;'>"

        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                chat_html += f"""
                <div style="background-color:#e3f2fd; padding:10px; border-radius:8px; 
                            margin:8px 0; border-left:4px solid #2196f3;">
                    <strong>🙋 Siz:</strong> {message['content']}
                </div>
                """
            else:
                chat_html += f"""
                <div style="background-color:#f1f8e9; padding:10px; border-radius:8px; 
                            margin:8px 0; border-left:4px solid #4caf50;">
                    <strong>🤖 Asistan:</strong> {message['content']}
                </div>
                """

        chat_html += "</div>"
        chat_html += """
        <script>
            var chatBox = document.getElementById('chat-box');
            chatBox.scrollTop = chatBox.scrollHeight;
        </script>
        """
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="height: 150px; border: 2px dashed #ccc; border-radius: 10px; 
                        display: flex; align-items: center; justify-content: center; 
                        color: #666; margin-bottom: 20px;">
                <p>💭 Sohbet henüz başlamadı. Bir fotoğraf yükleyin ve soru sormaya başlayın!</p>
            </div>
            """,
            unsafe_allow_html=True
        )
