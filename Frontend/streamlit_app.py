import streamlit as st
pages = {
    "Principal": [
        st.Page("homepage.py", title="Tomate ResNet"),
        st.Page("test_resnet.py", title="Prueba de Clasificación"),
    ],
    "Explicaciones": [
        st.Page("pgc_presentation.py", title="PGG Información"),
        st.Page("resnet_architecture.py", title="ResNet Arquitectura"),
    ],
    "Código": [
        st.Page("dataset_explanation.py", title="Explicación del Dataset"),
        st.Page("tomato_categories_explanation.py", title="Explicación de las Categorías"),
        st.Page("documentation.py", title="Documentación"),
    ],
}

pg = st.navigation(pages)
pg.run()

logo_path = "Resources/Images/tomate_logo_large.png"

st.logo(logo_path, size="large")