import streamlit as st
import google.generativeai as genai
from PIL import Image
from get_prediction import predict_image

# Configure page
st.set_page_config(
    page_title="Beslenme Analiz Uygulaması",
    page_icon="🍎",
    layout="wide"
)

# Title and description
st.title("🍎 Yiyecek Beslenme Analizi")
st.markdown("Detaylı beslenme bilgisi almak için bir yiyecek fotoğrafı yükleyin!")

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
        st.image(image, caption="Yüklenen Fotoğraf", use_column_width=True)

with col2:
    st.header("📊 Beslenme Analizi")

    if uploaded_file is not None and gemini_api_key:
        try:
            with st.spinner("Fotoğraf analiz ediliyor..."):
                # Process image with your SigLIP2 regressor
                # Note: You'll need to import your predict_image function
                weight, cal, carb, fat, protein = predict_image(uploaded_file).squeeze().tolist()

            # Display raw predictions
            st.subheader("🔢 Tespit Edilen Değerler")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

            with metrics_col1:
                st.metric("Ağırlık", f"{weight:.1f}g")
                st.metric("Kalori", f"{cal:.0f} kcal")

            with metrics_col2:
                st.metric("Karbonhidrat", f"{carb:.1f}g")
                st.metric("Yağ", f"{fat:.1f}g")

            with metrics_col3:
                st.metric("Protein", f"{protein:.1f}g")

            # Generate AI interpretation with caching
            # Create a unique key for this image and predictions
            prediction_key = f"{uploaded_file.name}_{weight:.1f}_{cal:.0f}_{carb:.1f}_{fat:.1f}_{protein:.1f}"

            if 'ai_response' not in st.session_state or st.session_state.get('last_prediction_key') != prediction_key:
                with st.spinner("AI öngörüleri alınıyor..."):
                    try:
                        model = genai.GenerativeModel('gemini-2.0-flash-exp')

                        # Prepare the image for Gemini
                        image_pil = Image.open(uploaded_file)

                        prompt = f"""
                        Bu fotoğrafta gördüğün yiyeceği tanımla ve aşağıdaki beslenme verilerini analiz et:

                        Ağırlık: {weight:.1f}g
                        Kalori: {cal:.0f} kcal
                        Karbonhidrat: {carb:.1f}g
                        Yağ: {fat:.1f}g
                        Protein: {protein:.1f}g

                        ÖNEMLİ: Yorumunu sadece verilen sayısal değerlere dayandır, fotoğraftaki görünümü sadece yiyecek türünü tanımlamak için kullan.

                        Lütfen şunları sağla:
                        1. Fotoğraftaki yiyeceğin ne olduğunu kısaca belirt
                        2. Verilen beslenme değerlerinin bu yiyecek için değerlendirmesi
                        3. Sağlık faydaları veya dikkat edilmesi gerekenler
                        4. Bu yiyeceğin dengeli bir diyete nasıl uyduğu
                        5. Dikkat çekici beslenme özelikleri
                        6. Varsa porsiyon boyutu önerileri

                        Cevabı Türkçe olarak ver. Bilgilendirici ama anlaşılır tut, yaklaşık 250-350 kelime.
                        """

                        response = model.generate_content([prompt, image_pil])

                        # Cache the response
                        st.session_state['ai_response'] = response.text
                        st.session_state['last_prediction_key'] = prediction_key

                    except Exception as e:
                        st.error(f"AI öngörüleri alınırken hata: {str(e)}")
                        st.info("Ham beslenme verileri yukarıda hala mevcut.")
                        st.session_state['ai_response'] = None

            # Display cached response
            if st.session_state.get('ai_response'):
                st.subheader("🤖 AI Beslenme Öngörüleri")
                st.write(st.session_state['ai_response'])

            # Additional visualizations
            st.subheader("📈 Beslenme Dağılımı")

            # Macronutrient pie chart
            import plotly.express as px
            import pandas as pd

            # Calculate calories from macronutrients (approximate)
            carb_cal = carb * 4
            protein_cal = protein * 4
            fat_cal = fat * 9

            macro_df = pd.DataFrame({
                'Makrobesin': ['Karbonhidrat', 'Protein', 'Yağ'],
                'Kalori': [carb_cal, protein_cal, fat_cal],
                'Gram': [carb, protein, fat]
            })

            fig = px.pie(macro_df, values='Kalori', names='Makrobesin',
                         title="Makrobesinlere Göre Kalorik Dağılım")
            st.plotly_chart(fig, use_container_width=True)

            # Nutritional density bar chart
            density_df = pd.DataFrame({
                'Besin': ['Karbonhidrat', 'Protein', 'Yağ'],
                '100g başına': [carb / weight * 100, protein / weight * 100, fat / weight * 100]
            })

            fig2 = px.bar(density_df, x='Besin', y='100g başına',
                          title="Besin Yoğunluğu (100g başına gram)")
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Fotoğraf işlenirken hata: {str(e)}")
            st.info(
                "Lütfen predict_image fonksiyonunuzun doğru şekilde import edildiğinden ve fotoğrafın geçerli olduğundan emin olun.")

    elif uploaded_file is not None and not gemini_api_key:
        st.warning("⚠️ AI öngörüleri almak için lütfen kenar çubuğuna Gemini API anahtarınızı girin.")

    elif not uploaded_file:
        st.info("👆 Analizi başlatmak için lütfen bir fotoğraf yükleyin.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <p>SigLIP2 Regressor & Google Gemini 2.0 Flash ile güçlendirilmiştir</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Instructions for setup
with st.expander("📋 Kurulum Talimatları"):
    st.markdown("""
    **Bu uygulamayı kullanmak için:**

    1. **Gerekli paketleri yükleyin:**
       ```bash
       pip install streamlit google-generativeai pillow plotly pandas
       ```

    2. **Gemini API anahtarı alın:**
       - [Google AI Studio](https://aistudio.google.com/app/apikey) adresine gidin
       - Yeni bir API anahtarı oluşturun
       - Kenar çubuğuna girin

    3. **Modelinizi import edin:**
       - `predict_image` fonksiyonunuzun mevcut olduğundan emin olun
       - Üst kısma şu import'u ekleyin: `from your_model_file import predict_image`

    4. **Uygulamayı çalıştırın:**
       ```bash
       streamlit run app.py
       ```
    """)