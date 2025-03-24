import streamlit as st
import base64
st.set_page_config(layout="wide")

st.title(":primary[Red Neuronal Convolucional para la Clasificación de Hojas Sanas y Enfermas en Plantas de Tomate]")

st.write("""
            ## **Integrantes:**
            ### **John Sebastián Galindo Hernández**
            ### **Juan David Moreno Beltrán**
            ### **Miguel Ángel Moreno Beltrán**
         
         """)

# Github icon

st.markdown("<br> <br>", unsafe_allow_html=True)


st.markdown(
    """<a href="https://github.com/">
    <img src="data:image/png;base64,{}" width="50">
    </a>""".format(
        base64.b64encode(open("Resources/Images/github-mark.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)