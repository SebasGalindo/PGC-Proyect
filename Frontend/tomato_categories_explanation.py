import streamlit as st
categories = {
                "Tomate con mancha bacteriana":
                    {
                    "Descripción":
                        "La mancha bacteriana se manifiesta en las hojas con pequeñas manchas oscuras, inicialmente de apariencia aceitosa, que luego se vuelven crujientes y amarillentas; estas lesiones pueden unirse y causar la caída prematura de las hojas, debilitando la planta. En los frutos verdes, la mancha bacteriana se presenta como pequeñas protuberancias similares a ampollas con un halo amarillento.",
                    "link": "https://www.thespruce.com/identify-treat-prevent-tomato-diseases-7153094",
                    "Imagen": "Resources/Images/Bacterial_Spot.png"
                    },
                "Tomate con tizón temprano":
                    {
                    "Descripción":
                        " El tizón temprano se identifica por la aparición de manchas oscuras en las hojas más viejas, caracterizadas por anillos concéntricos como los de una diana, rodeadas de un halo amarillo; estas lesiones se expanden con el tiempo, provocando que las hojas se vuelvan amarillas, quebradizas y finalmente se caigan, exponiendo los frutos a un sol excesivo si las hojas superiores también se ven afectadas.",
                    "link": " https://www.thespruce.com/early-blight-on-tomato-plants-1402973",
                    "Imagen": "Resources/Images/Early_Blight.JPG"
                    },
                "Tomate con tizón tardío":
                    {
                    "Descripción":
                        "El tizón tardío se reconoce por la presencia de manchas grandes e irregulares de color verde grisáceo en las hojas, que rápidamente se oscurecen y pueden desarrollar un moho blanquecino en la parte inferior de la hoja en condiciones de alta humedad; las hojas infectadas se marchitan, se enrollan hacia arriba y mueren, dando a la planta una apariencia quemada, siendo más común en climas frescos y lluviosos.",
                    "link": "https://www.thespruce.com/identify-treat-prevent-tomato-diseases-7153094",
                    "Imagen": "Resources/Images/Late_Blight.JPG"
                    },
                "Tomate con moho foliar":
                    {
                    "Descripción":
                        "El moho foliar se caracteriza por la aparición de un polvo blanco o grisáceo que cubre la superficie superior de las hojas, comenzando como parches aislados y extendiéndose hasta que el follaje se torna amarillo, se enrolla hacia arriba y finalmente se cae, afectando primero a las hojas más maduras y pudiendo extenderse a tallos y flores.",
                    "link": "https://www.thespruce.com/identify-treat-prevent-tomato-diseases-7153094",
                    "Imagen": "Resources/Images/Leaf_Mold.JPG"
                    },
                "Tomate con mancha foliar de Septoria":
                    {
                    "Descripción":
                        "La mancha foliar por Septoria se identifica por pequeñas manchas circulares de color gris claro con bordes oscuros y pequeños puntos negros en el centro de la lesión; estas manchas se multiplican, causando un amarillamiento generalizado que comienza en las hojas inferiores y avanza hacia arriba, pudiendo dejar la planta casi sin follaje.",
                    "link": "https://www.thespruce.com/identify-treat-prevent-tomato-diseases-7153094",
                    "Imagen": "Resources/Images/Septoria_Spot.JPG"
                    },
                "Tomate con ácaro araña de dos manchas":
                    {
                    "Descripción":
                        "Los ácaros araña roja de dos puntos en los tomates causan inicialmente pequeños puntos amarillos o blancos en las hojas, dándoles un aspecto punteado o con picaduras; a medida que la infestación avanza, estos puntos pueden unirse y formar áreas más grandes de color amarillo o bronceado, y en la parte inferior de las hojas, a menudo se puede observar una fina tela de araña junto con los diminutos ácaros de color rojizo. En casos severos, las hojas pueden enrollarse, secarse y caerse prematuramente, comenzando generalmente por las hojas más bajas de la planta.",
                    "link": "https://extension.umd.edu/resource/key-common-problems-tomatoes/",
                    "Imagen": "Resources/Images/Two-Spotted_Spider_Mite.JPG"
                    },
                "Tomate con mancha de diana":
                    {
                    "Descripción":
                        "La mancha diana se caracteriza por la aparición en las hojas de manchas circulares con anillos concéntricos de color marrón y amarillo, similares a una diana; estas lesiones crecen y pueden agrietarse o perforarse en etapas avanzadas, dando al follaje un aspecto de 'encaje', afectando primero a las hojas inferiores y ascendiendo en condiciones húmedas.",
                    "link": "https://www.thespruce.com/identify-treat-prevent-tomato-diseases-7153094",
                    "Imagen": "Resources/Images/Target_Spot.JPG"
                    },
                "Tomate con virus del mosaico":
                    {
                    "Descripción":
                        "El virus del mosaico se evidencia en las hojas por un patrón de mosaico con áreas de color verde claro y oscuro, acompañado de deformaciones como rizado, alargamiento o reducción del tamaño de la hoja; el follaje joven puede verse arrugado y frágil, mientras que las hojas más viejas desarrollan parches amarillos irregulares, resultando en un crecimiento desigual y débil de la planta.",
                    "link": "https://www.thespruce.com/identify-treat-prevent-tomato-diseases-7153094",
                    "Imagen": "Resources/Images/Mosaic_Virus.JPG"
                    },
                "Tomate con virus del rizo amarillo":
                    {
                    "Descripción":
                        "El virus del rizo amarillo provoca que las hojas nuevas se rizan hacia arriba, adoptando una forma de cuchara, y presenten un amarillamiento en los bordes mientras el centro permanece verde; las hojas afectadas son más pequeñas y rígidas, y la planta detiene su crecimiento vertical, adquiriendo un aspecto arbustivo y atrofiado, pudiendo ocurrir una caída prematura del follaje.",
                    "link": "https://www.thespruce.com/identify-treat-prevent-tomato-diseases-7153094",
                    "Imagen": "Resources/Images/Yellow_Curl_Virus.JPG"
                    },
                "Tomate saludable":
                    {
                    "Descripción":
                        "Un tomate saludable presenta hojas de un color verde uniforme y brillante, sin manchas, decoloraciones o deformaciones; su crecimiento es equilibrado, con tallos firmes y un follaje denso pero aireado, y las hojas jóvenes se desarrollan sin rizarse ni mostrar lesiones sospechosas.",
                    "link": None,
                    "Imagen": "Resources/Images/Healthy.JPG"
                    },
    }


st.set_page_config(layout="wide")
st.title(":primary[Explicación de las Categorías]")

for i, category, value in zip(range(10), categories.keys(), categories.values()):
    
    col1, col2, col3 = st.columns([2,5,1])
    st.divider()
    col1.write("")
    col1.image(value["Imagen"], use_container_width=True)
    col2.markdown(f"#### :primary[{category}]")
    col2.markdown(f"##### {value['Descripción']}")
    if value["link"] is not None:
        col2.caption(f"Fuente: {value['link']}")

