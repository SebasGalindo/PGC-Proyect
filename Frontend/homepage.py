import streamlit as st
import base64
st.set_page_config(layout="wide")

st.title(":primary[Red Neuronal Convolucional para la Clasificación de Hojas Sanas y Enfermas en Plantas de Tomate]")

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

col1.write("""
            ## **Integrantes:**
            ### **John Sebastián Galindo Hernández**
            ### **Juan David Moreno Beltrán**
            ### **Miguel Ángel Moreno Beltrán**
         
         """)

#Qr image for test the app
col2.image(
    "Resources/Images/qr_pgc_code.png",
    caption="Escanea el código QR para probar la aplicación.",
    use_container_width=False,
)

# Github icon
st.markdown(
    """<a href="https://github.com/">
    <img src="data:image/png;base64,{}" width="50">
    </a>""".format(
        base64.b64encode(open("Resources/Images/github-mark.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)

