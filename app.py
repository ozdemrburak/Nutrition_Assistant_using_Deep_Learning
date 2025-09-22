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
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Fotoğraf", use_column_width=50)

        if gemini_api_key:
            try:
                with st.spinner("Fotoğraf analiz ediliyor..."):
                    # Process image with SigLIP2 regressor - Fixed variable order
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

        # Scrollable chat container
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                chat_area = st.container()
                with chat_area:
                    # Scrollable box için CSS
                    st.markdown(
                        """
                        <style>
                        .scroll-box {
                            max-height: 400px;
                            overflow-y: auto;
                            border: 1px solid #e0e0e0;
                            border-radius: 10px;
                            padding: 15px;
                            background-color: #fafafa;
                        }
                        .user-msg {
                            background-color: #e3f2fd;
                            padding: 10px;
                            border-radius: 8px;
                            margin: 8px 0;
                            border-left: 4px solid #2196f3;
                        }
                        .assistant-msg {
                            background-color: #f1f8e9;
                            padding: 10px;
                            border-radius: 8px;
                            margin: 8px 0;
                            border-left: 4px solid #4caf50;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown('<div class="scroll-box">', unsafe_allow_html=True)
                    for i, message in enumerate(st.session_state.chat_history):
                        if message['role'] == 'user':
                            st.markdown(f'<div class="user-msg"><strong>🙋 Siz:</strong> {message["content"]}</div>',
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f'<div class="assistant-msg"><strong>🤖 Asistan:</strong> {message["content"]}</div>',
                                unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
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

    # Chat input
    if st.session_state.current_analysis and gemini_api_key:
        user_question = st.text_input(
            "Beslenme hakkında soru sorun:",
            placeholder="Örnek: Bu yiyecek kaç kişilik? Diyetime uygun mu? Hangi besinler eksik?",
            key="user_input"
        )

        col_send, col_examples = st.columns([1, 2])

        with col_send:
            if st.button("📨 Gönder") and user_question:
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_question
                })

                # Generate response
                try:
                    with st.spinner("Cevap hazırlanıyor..."):
                        model = genai.GenerativeModel('gemini-2.5-flash')

                        # Prepare context
                        analysis = st.session_state.current_analysis
                        context = f"""
                        Kullanıcının yüklediği yiyecek hakkında şu veriler var:
                        Ağırlık: {analysis['weight']:.1f}g
                        Kalori: {analysis['calories']:.0f} kcal
                        Karbonhidrat: {analysis['carbs']:.1f}g
                        Yağ: {analysis['fat']:.1f}g
                        Protein: {analysis['protein']:.1f}g

                        Sohbet geçmişi:
                        {chr(10).join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-5:]])}

                        Kullanıcı sorusu: {user_question}

                        Sadece bu beslenme verilerine dayanarak cevap ver. Kısa ve anlaşılır ol (100-200 kelime). Türkçe cevapla.
                        """

                        # Include image for better context
                        image_pil = Image.open(analysis['image'])
                        response = model.generate_content([context, image_pil])

                        # Add response to history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response.text
                        })

                        # Clear input and rerun
                        st.rerun()

                except Exception as e:
                    st.error(f"Cevap alınırken hata: {str(e)}")

        with col_examples:
            st.markdown("**💡 Örnek sorular:**")
            example_questions = [
                "Bu yiyecek sağlıklı mı?",
                "Kaç kişilik porsiyon?",
                "Hangi vitaminler var?",
                "Diyetime uygun mu?",
                "Kalori yoğunluğu nasıl?"
            ]

            for idx, question in enumerate(example_questions):
                if st.button(question, key=f"example_{idx}"):
                    # Add user message to history
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question
                    })

                    # Generate response
                    try:
                        with st.spinner("Cevap hazırlanıyor..."):
                            model = genai.GenerativeModel('gemini-2.5-flash')

                            # Prepare context
                            analysis = st.session_state.current_analysis
                            context = f"""
                            Kullanıcının yüklediği yiyecek hakkında şu veriler var:
                            Ağırlık: {analysis['weight']:.1f}g
                            Kalori: {analysis['calories']:.0f} kcal
                            Karbonhidrat: {analysis['carbs']:.1f}g
                            Yağ: {analysis['fat']:.1f}g
                            Protein: {analysis['protein']:.1f}g

                            Kullanıcı sorusu: {question}

                            Sadece bu beslenme verilerine dayanarak cevap ver. Kısa ve anlaşılır ol (100-200 kelime). Türkçe cevapla.
                            """

                            # Include image for better context
                            image_pil = Image.open(analysis['image'])
                            response = model.generate_content([context, image_pil])

                            # Add response to history
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': response.text
                            })

                            # Rerun to show new messages
                            st.rerun()

                    except Exception as e:
                        st.error(f"Cevap alınırken hata: {str(e)}")
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': f"Üzgünüm, bir hata oluştu: {str(e)}"
                        })
                        st.rerun()

    elif not st.session_state.current_analysis:
        st.info("👆 Sohbet etmek için önce bir fotoğraf yükleyin ve analiz edin.")
    elif not gemini_api_key:
        st.warning("⚠️ Sohbet için API anahtarınızı girin.")

# Visualization section (if analysis exists) - in a scrollable container
if st.session_state.current_analysis:
    st.markdown("---")
    st.header("📈 Görsel Analiz")

    # Create scrollable container for visualizations
    st.markdown(
        """
        <div style="max-height: 600px; overflow-y: auto; border: 1px solid #e0e0e0; 
                    border-radius: 10px; padding: 15px; background-color: #fafafa;">
        """,
        unsafe_allow_html=True
    )

    analysis = st.session_state.current_analysis

    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        # Macronutrient pie chart
        import plotly.express as px
        import pandas as pd

        carb_cal = analysis['carbs'] * 4
        protein_cal = analysis['protein'] * 4
        fat_cal = analysis['fat'] * 9

        macro_df = pd.DataFrame({
            'Makrobesin': ['Karbonhidrat', 'Protein', 'Yağ'],
            'Kalori': [carb_cal, protein_cal, fat_cal]
        })

        fig = px.pie(macro_df, values='Kalori', names='Makrobesin',
                     title="Kalorik Dağılım")
        st.plotly_chart(fig, use_container_width=True)

    with col_viz2:
        # Nutritional density bar chart
        density_df = pd.DataFrame({
            'Besin': ['Karb.', 'Protein', 'Yağ'],
            '100g başına': [
                analysis['carbs'] / analysis['weight'] * 100,
                analysis['protein'] / analysis['weight'] * 100,
                analysis['fat'] / analysis['weight'] * 100
            ]
        })

        fig2 = px.bar(density_df, x='Besin', y='100g başına',
                      title="Besin Yoğunluğu")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <p>Custom SigLIP2 Regressor & Google Gemini 2.5 Flash ile güçlendirilmiştir</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Instructions
with st.expander("📋 Nasıl Kullanılır"):
    st.markdown("""
    **Adım adım kullanım:**

    1. **API Anahtarı**: Kenar çubuğuna Gemini API anahtarınızı girin
    2. **Fotoğraf Yükle**: Sol taraftan bir yiyecek fotoğrafı seçin
    3. **İlk Analiz**: Sistem otomatik olarak beslenme analizini yapar
    4. **Soru Sor**: Sağ taraftaki chatbot'a istediğiniz soruyu sorun
    5. **Detaylı Bilgi**: Örnek sorulardan seçebilir veya kendi sorunuzu yazabilirsiniz

    **API Anahtarı almak için:**
    - [Google AI Studio](https://aistudio.google.com/app/apikey) adresine gidin
    - Yeni bir API anahtarı oluşturun
    """)