import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title(":primary[Explicación del Dataset]")

st.markdown("#### El dataset utilizado para el desarrollo de este proyecto es el PlantVillage Dataset, el cual contiene imágenes de hojas de tomate sanas y enfermas.")
st.markdown("#### Este dataset se compone de gran cantidad de imagenes en tres tipos diferentes: Imagenes a color y con fondo, Imagenes segmentadas sin fondo e Imagenes a blanco y negro, todas con una resolución de :primary[256 x 256 píxeles]")
st.markdown("#### Para este proyecto se requerian solo las imagenes de tomate y se decidió usar las imagenes a color quedando utilizables :primary[18.160] imagenes entre 10 diferentes categorias:")
st.table(
    pd.DataFrame(
        {
            "Categoría": [
                "Tomate con mancha bacteriana",
                "Tomate con tizón temprano",
                "Tomate con tizón tardío",
                "Tomate con moho foliar",
                "Tomate con mancha foliar de Septoria",
                "Tomate con ácaro araña de dos manchas",
                "Tomate con mancha de diana",
                "Tomate con virus del mosaico",
                "Tomate con virus del rizo amarillo",
                "Tomate saludable"
            ],
            "Cantidad de imagenes":[2127,1000,1909,1400,1771,1676,1235,373,5359,1591]
        }
    )
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### :primary[**Fuente:**] Kaggle")
    
with col2:
    st.markdown("#### :primary[**Link:**] [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)")
    
    
st.markdown("#### Como se decidió trabajar con ResNet50 se requirió reestructurar el dataset para que cumpliera con los requerimientos de la red neuronal quedando con la siguiente estructura:")

st.markdown(":material/folder: :primary[**Tomato-Dataset**]")
st.markdown("|$  \\quad$ :material/folder: :primary[**Train**]")
st.markdown("|$  \\quad$ | $\\quad$ :material/folder: :primary[**Saludable**]")
st.markdown("|$  \\quad$ | $\\quad$ | $\\quad$ :material/image: :primary[**imagenes.jpg**]")
st.markdown("|$  \\quad$ | $\\quad$ :material/folder: :primary[**Virus_Mosaico**]")
st.markdown("|$  \\quad$ | $\\quad$ | $\\quad$ :material/image: :primary[**imagenes.jpg**]")
st.markdown("|$  \\quad$ | $\\quad$ :material/folder: :primary[**Categoría...**]")
st.markdown("|$  \\quad$ | $\\quad$ | $\\quad$ :material/image: :primary[**imagenes.jpg**]")
st.markdown("|$  \\quad$ :material/folder: :primary[**Validation**]")
st.markdown("|$  \\quad$ | $\\quad$ :material/folder: :primary[**Saludable**]")
st.markdown("|$  \\quad$ | $\\quad$ | $\\quad$ :material/image: :primary[**imagenes.jpg**]")
st.markdown("|$  \\quad$ | $\\quad$ :material/folder: :primary[**Virus_Mosaico**]")
st.markdown("|$  \\quad$ | $\\quad$ | $\\quad$ :material/image: :primary[**imagenes.jpg**]")
st.markdown("|$  \\quad$ | $\\quad$ :material/folder: :primary[**Categoría...**]")
st.markdown("|$  \\quad$ | $\\quad$ | $\\quad$ :material/image: :primary[**imagenes.jpg**]")
st.markdown("|$  \\quad$ :material/folder: :primary[**Test**]")
st.markdown("|$  \\quad$ | $\\quad$ :material/folder: :primary[**Saludable**]")
st.markdown("|$  \\quad$ | $\\quad$ | $\\quad$ :material/image: :primary[**imagenes.jpg**]")
st.markdown("|$  \\quad$ | $\\quad$ :material/folder: :primary[**Virus_Mosaico**]")
st.markdown("|$  \\quad$ | $\\quad$ | $\\quad$ :material/image: :primary[**imagenes.jpg**]")
st.markdown("|$  \\quad$ | $\\quad$ :material/folder: :primary[**Categoría...**]")
st.markdown("|$  \\quad$ | $\\quad$ | $\\quad$ :material/image: :primary[**imagenes.jpg**]")

st.markdown("#### Para la reestructuración del dataset se utilizó el script [:primary['suffle-data']](https://www.kaggle.com/emmarex/plantdisease) (ubicado en el repositorío)")




