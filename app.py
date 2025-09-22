import streamlit as st

st.set_page_config(page_title="Beslenme Asistanı", layout="wide")

# Başlık
st.title("🥗 Beslenme Asistanı")

# Session state başlat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Örnek resim + değerler (normalde modelden veya veri tabanından gelecek)
food_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/480px-No_image_available.svg.png"
nutrition_info = {
    "Kalori": "208 kcal",
    "Protein": "12.8 g",
    "Karbonhidrat": "20.1 g",
    "Yağ": "7.9 g"
}

# Resim + değerler üstte
with st.container():
    cols = st.columns([2, 3])
    with cols[0]:
        st.image(food_image_url, width=150, caption="Yemek Görseli")
    with cols[1]:
        st.markdown("### Besin Değerleri")
        st.markdown(
            f"""
            - **Kalori:** {nutrition_info['Kalori']}  
            - **Protein:** {nutrition_info['Protein']}  
            - **Karbonhidrat:** {nutrition_info['Karbonhidrat']}  
            - **Yağ:** {nutrition_info['Yağ']}  
            """
        )

st.markdown("---")

# Sohbet ekranı (scrollable)
if st.session_state.chat_history:
    with st.container():
        chat_box = ""
        chat_box += "<div id='chat-box' style='height:400px; overflow-y:auto; border:1px solid #ddd; border-radius:10px; padding:10px; background:#fafafa;'>"
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_box += f"""
                <div style="background-color:#e3f2fd; padding:8px; border-radius:6px;
                            margin:6px 0; border-left:4px solid #2196f3;">
                    <strong>🙋 Siz:</strong> {msg['content']}
                </div>
                """
            else:
                chat_box += f"""
                <div style="background-color:#f1f8e9; padding:8px; border-radius:6px;
                            margin:6px 0; border-left:4px solid #4caf50;">
                    <strong>🤖 Asistan:</strong> {msg['content']}
                </div>
                """
        chat_box += "</div>"

        # Otomatik scroll en alta
        chat_box += """
        <script>
            var chatBox = document.getElementById('chat-box');
            chatBox.scrollTop = chatBox.scrollHeight;
        </script>
        """

        st.markdown(chat_box, unsafe_allow_html=True)

# Input alanı
user_input = st.text_input("Beslenme hakkında soru sorun:")

if user_input:
    # Kullanıcı mesajı ekle
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    # Asistan cevabı (dummy, burada model bağlanacak)
    if "kaç kişilik" in user_input.lower():
        response = "📌 Bu porsiyon tek kişilik kabul edilebilir."
    else:
        response = "Bu yiyecek dengeli bir besin profiline sahip görünüyor."
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.experimental_rerun()
