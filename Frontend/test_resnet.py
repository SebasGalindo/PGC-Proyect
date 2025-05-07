import streamlit as st
import requests
from image_uploader import image_uploader
import base64
st.set_page_config(layout="wide")
st.title(":primary[Tomato ResNet]")

st.write("Esta es una  aplicación web que clasifica imágenes de las hojas en la planta de tomates en diez categorías diferentes.")
st.write("Cada categoría se representa con un número del 0 al 9.")

categories = {
                "Mancha_Bacteriana":"Tomate con mancha bacteriana",
                "Tizon_Temprano":"Tomate con tizón temprano",
                "Tizon_Tardio":"Tomate con tizón tardío",
                "Moho_Foliar":"Tomate con moho foliar",
                "Mancha_Foliar_Por_Septoria":"Tomate con mancha foliar de Septoria",
                "Acaro_Araña_Roja_Dos_Puntos":"Tomate con ácaro araña de dos manchas",
                "Mancha_Diana":"Tomate con mancha de diana",
                "Virus_Mosaico":"Tomate con virus del mosaico",
                "Virus_rizado_amarillo":"Tomate con virus del rizo amarillo",
                "Saludable":"Tomate saludable"
    }

with st.expander("Posibles Categorías "):
    values = list(categories.values())
    for categorie in values:
        st.write(f"**-> $\\quad$ {categorie}**")
        
st.write("""### :primary[ **Prueba de clasificación** ]""")
image_label  =  "Por favor, sube una imagen de una hoja de tomate para clasificarla."
    
result = image_uploader(
        buttonText="Subir Imagen",
        dropText="Arrastra y Suelta la Imagen Aquí",
        allowedFormatsText="Allowed formats: .jpg, .png",
        borderColor="#1c5938",
        buttonColor="#efddce",
        buttonTextColor="#1c5938",
        hoverButtonColor="#ECBF9A",
        key="1"
    )

preview = result.get("preview") if result is not None else None

if preview is not None:
    data_base64 = preview.split(",")[1]

    # Decodificamos la cadena base64 a bytes
    image = base64.b64decode(data_base64)
    
    # mandar la imagen al backend
    url = "https://sebastian-galindo-tomato-resnet.hf.space/upload"
    files = {'image': image}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        category = response.json()["category"]
        st.write("### :primary[La imagen ha sido clasificada como:]")
        st.write(f" #### :orange[**Hoja de {categories[category]}**]")