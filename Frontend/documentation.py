import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# Título y descripción general
st.title("Documentación")
st.markdown("""
Esta aplicación sirve como documentación para el código de entrenamiento y prueba de un modelo ResNet50, 
diseñado para la clasificación de enfermedades en hojas de tomate. Aquí se detallan las librerías y dependencias 
utilizadas, el rol de cada una, y cómo se integran para lograr un proceso de entrenamiento, validación y prueba 
eficiente. Además, se explica el uso de Hugging Face para el despliegue del API y Plotly para una visualización 
más atractiva de la matriz de confusión.
""")

# Sección: Dependencias Generales
st.header("Dependencias Generales y su Función")
st.markdown("""
A continuación, se detalla cada dependencia y por qué es fundamental en el proceso:

- **os**: Se utiliza para el manejo de archivos y rutas en el sistema operativo. Es fundamental para acceder a los 
  datos de entrenamiento, validación y prueba almacenados en diferentes carpetas.
- **torch**: Es la librería principal de PyTorch, que permite la construcción y entrenamiento de modelos de deep learning.
- **torch.nn**: Proporciona las herramientas para definir módulos y capas de redes neuronales, permitiendo la construcción 
  del modelo y la definición de funciones de pérdida.
- **torchvision (datasets, transforms)**: Se utiliza para cargar y transformar los conjuntos de datos de imágenes. 
  Las transformaciones permiten ajustar y normalizar las imágenes (por ejemplo, redimensionamiento, normalización y 
  aumentos de datos como el flip horizontal).
- **torch.utils.data.DataLoader**: Facilita la creación de iteradores sobre los datasets, permitiendo el procesamiento 
  de imágenes en lotes (batches) y la paralelización con múltiples procesos.
- **timm**: Una librería que provee implementaciones de modelos preentrenados de vanguardia. En este caso se usa 
  para cargar el modelo ResNet50 con pesos preentrenados, lo cual acelera el proceso de entrenamiento al transferir 
  el aprendizaje.
- **time**: Utilizada para medir el tiempo de procesamiento de cada batch y de cada época, lo que permite monitorear 
  el rendimiento y el tiempo de entrenamiento.
- **collections.Counter**: Facilita el conteo de la frecuencia de las etiquetas en el dataset, ayudando a identificar 
  posibles desbalances en la distribución de las clases.
- **sklearn.metrics (confusion_matrix, classification_report)**: Se usan para generar la matriz de confusión y 
  el reporte de clasificación, herramientas esenciales para evaluar el desempeño del modelo durante las pruebas.
- **pandas**: Permite la manipulación y el almacenamiento de datos de forma tabular. En este caso, se utiliza para guardar 
  la matriz de confusión en formato CSV.
- **json**: Se emplea para guardar el reporte de clasificación en formato JSON, lo que facilita su lectura y posterior análisis.
""")
# region Explicacion del codigo principal
# Sección: Entrenamiento en un notebook de Kaggle para aprovechar las 2 GPUs T4 Gratis con 30 horas de uso semanales
st.header("Entrenamiento en un Notebook de Kaggle")
st.markdown("""
Para el entrenamiento del modelo ResNet50, se recomienda utilizar un entorno con aceleración por GPU para acelerar el
proceso. En este caso, se optó por un notebook de Kaggle, que ofrece acceso gratuito a GPUs T4 con 16 GB de VRAM. lo que permitió
majear tiempos de entrenamiento más cortos aproximados a 2 minutos por época. Además, Kaggle proporciona 30 horas de uso
semanal de GPU, lo que permite realizar múltiples experimentos y ajustes de hiperparámetros sin incurrir en costos adicionales.
""")

st.markdown("**El siguiete codigo especifica el uso de GPU y la cantidad a usar:**")
gpu_code = """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
"""
st.code(gpu_code, language="python")

st.markdown("**Para el entrenamiento del modelo se necesitaban las siguientes funciones:**")
st.caption("aunque el codigo en kaggle se encuentra en una sola funcion para facilitar el entendimiento se separaron en funciones las funcionalidades")
st.markdown("**1. Función de Transformaciones de Imágenes:**")
st.markdown("Esta función define las transformaciones necesarias para preprocesar las imágenes antes de ser alimentadas al modelo. (Redimiensionandolas y normalizandolas)")
image_transforms_code = """
def image_transforms():
    # 1. Define the transformations for the images
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)), # Resize the images to 224x224 pixels
        transforms.RandomHorizontalFlip(), # Random horizontal flip
        transforms.ToTensor(), # Convert the images to tensors (values between 0 and 1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) # Normalize the images
    ])

    # The same transformations are used for the validation and test datasets
    # The only difference is that the RandomHorizontalFlip transformation is not used because it is not necessary
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    return transform_train, transform_val
"""
st.code(image_transforms_code, language="python")

st.markdown("**2. Función de Carga de Datasets:**")
st.markdown("Esta función carga los datasets de entrenamiento, validación y prueba, aplicando las transformaciones definidas anteriormente.")

datasets_code = """
def load_datasets(train_path, val_path, test_path, transform_train, transform_val):
    # 2. Load the datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
    class_names = train_dataset.classes

    val_dataset = datasets.ImageFolder(root=val_path, transform=transform_val)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform_val)
    print("Datasets loaded")

    # set the train, validation and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    print("Data loaders set")
    return train_loader, val_loader, test_loader
    """
st.code(datasets_code, language="python")

st.markdown("**3. Función de Creación del Modelo:**")
st.markdown("Esta función define el modelo ResNet50, lo carga con los pesos preentrenados de ImageNet  y cambia la última capa para adaptarla al número de clases del dataset.")
model_code = """
def create_model():
    # 3. Define the ResNet50 model
    model = timm.create_model('resnet50', pretrained=True)
    print("Model defined")

    # 4. Modify the last layer of the model to have 10 output classes (one for each category)
    model.fc = nn.Linear(model.fc.in_features, len(categories))
    print("Last layer modified")

    return model
"""
st.code(model_code, language="python")

st.markdown("**4. Función de Entrenamiento del Modelo:**")
st.markdown("Esta función entrena el modelo, calculando la pérdida y actualizando los pesos de la red mediante backpropagation.")
st.markdown("Se especifican los hiperparámetros como el número de épocas, la función de pérdida (CrossEntropyLoss) y el optimizador (Adam).")
train_model_code = """
def train_model(model, train_loader, val_loader, epochs=10):
    
    # 5. Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Loss function and optimizer defined")
    
    # 6. Train the model
    model, device = set_model_device(model)
    print(f"Training on {device}")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
        
    print("--------------------------------------------------------------------")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs} ")
        i = 0
        epoch_time = 0
        for images, labels in train_loader:
            i += 1
            actual_time = time.time()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            time_spend_in_sec = time.time()-actual_time
            epoch_time += time_spend_in_sec
            print(f'\r Batch {i} Process Time {round(time_spend_in_sec, 2)} seg,  Epoch Total Time: {round(epoch_time, 2)} seg', end=" ", flush=True)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Loss: {epoch_loss:.4f}")

        # 7. Validate the model
        model.eval()
        val_loss = 0.0
        val_correct = 0
        print("Validating...")
        with torch.no_grad():
            i = 0
            for images, labels in val_loader:
                i += 1
                actual_time = time.time()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
                time_spend_in_sec = time.time()-actual_time

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    print("Training completed.")
    return model
"""
st.code(train_model_code, language="python")

st.markdown("**5. Función de Evaluación del Modelo:**")
st.markdown("Esta función evalúa el modelo en el conjunto de prueba.")

test_model_code = """
def test_model_with_loader(model, test_loader, criterion):
    model, device = set_model_device(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
        
     # 8. Test the model
    model.eval()
    test_correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct.double() / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    """
st.code(test_model_code, language="python")

st.markdown("**6. Función de Guardado del Modelo:**")
st.markdown("Esta función guarda el modelo entrenado en un archivo .pth para su uso posterior.")
save_model_code = """
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f
"""
st.code(save_model_code, language="python")

st.markdown("**7. Función de Prueba del Modelo con una Imagen:**")
st.markdown("Esta función prueba el modelo con una imagen específica, devolviendo la predicción de la clase a la que pertenece.")
test_model_with_image_code = """
def test_model_with_image(model, image, transform_val):
    model, device = set_model_device(model)
    model.eval()
    with torch.no_grad():
        image = transform_val(image).unsqueeze(0).to(device)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        return preds.item()
"""
st.code(test_model_with_image_code, language="python")

st.markdown("**8. Función de Carga de un Modelo Preentrenado:**")
st.markdown("Esta función carga un modelo preentrenado desde un archivo .pth. y se encarga de convertir su información para que sea compatible con CPU o GPU.")
load_model_code = """
def load_model(path = "..\\Resources\\TomatoResNet50.pth"):
    model = create_model()
    model, device = set_model_device(model)
    state_dict = torch.load(path, map_location=device)
    if str(device) == "cpu" or (device == "cuda" and torch.cuda.device_count() == 1):
        print("removing module.")
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key.startswith("module."):
                new_key = key[len("module."):]
            else:
                new_key = key
            new_state_dict[new_key] = val

    model.load_state_dict(new_state_dict)
    return model
"""
st.code(load_model_code, language="python")

st.markdown("**9. Función de Configuración de Dispositivo y Modelo:**")
st.markdown("Esta función configura el dispositivo (CPU o GPU) y mueve el modelo a dicho dispositivo.")
set_model_device_code = """
  def set_model_device(model):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.to(device)
      if torch.cuda.device_count() > 1:
          model = nn.DataParallel(model)
          print(f"Using {torch.cuda.device_count()} GPUs")
      return model, device
"""
st.code(set_model_device_code, language="python")

# endregion
# region: Despliegue del API con Hugging Face
# Sección: Despliegue del API con Hugging Face
st.header("Despliegue del API con Hugging Face")
st.markdown("""
Para la parte de despliegue, se utiliza la plataforma **Hugging Face**. Esta plataforma permite alojar y exponer 
modelos de machine learning a través de APIs de forma sencilla. En el contexto de este proyecto:
- Se despliega un API que corre las pruebas del modelo, permitiendo enviar imágenes y recibir las predicciones del modelo.
- Hugging Face ofrece una infraestructura robusta y escalable, facilitando el acceso al modelo desde cualquier aplicación o servicio.
- La integración con Hugging Face posibilita realizar pruebas en tiempo real y monitorizar el desempeño del modelo en producción.
""")

st.markdown("El modelo se desplegó en los **Spaces** en un contenedor de docker con la version gratuita de 16GB de RAM ya que al ser solo para pruebas no es necesario las GPUs")
st.markdown("**El siguiente código muestra el dockerfile para el despliegue del modelo en Hugging Face:**")
dockerfile_code = """
FROM python:3.13.2-slim

# Configura las variables de caché
ENV XDG_CACHE_HOME=/app/.cache
ENV HF_HOME=/app/.cache/huggingface

# 1) Crear un usuario "user" (UID=1000) con su carpeta home
RUN useradd -m -u 1000 user

# 2) Crea /app, asígnale al usuario
RUN mkdir /app && chown user:user /app

# 3) Cambia el directorio de trabajo a /app
WORKDIR /app

# 4) Pasa a usar el usuario "user" (no root)
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
           
# 5) Copia el requirements.txt con los permisos correctos y lo instalas
COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# 6) Crea la carpeta de caché y dale permisos
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# 7) Copia todo tu código (incluyendo Resources/) a /app
COPY --chown=user:user . /app

# 8) Expón el puerto 7860 (usado por Hugging Face Spaces)
EXPOSE 7860

# 9) Inicia la app con Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "TomatoAPI:app"]
"""
st.code(dockerfile_code, language="docker")

st.markdown("**También se uso el siguiente dockerignore para evitar que el contenedor pesara demasiado:**")
dockerignore_code = """
# Ignorar entornos virtuales
venv/
.venv/
env/
ENV/
env.bak/
venv.bak/
ENV.bak/

# Ignorar archivos y carpetas de Python cacheados
__pycache__/
*.py[cod]
*$py.class

# Ignorar archivos de configuración de IDEs
.vscode/
.idea/

# Ignorar logs
*.log

# Ignorar archivos de compilación / distribución
build/
dist/
*.egg-info/
.eggs/
*.egg

# Ignorar archivos de configuración/estado temporales
*.swp
*~
"""
st.code(dockerignore_code, language="docker")

st.markdown("**El Proyecto tenia la siguiente estructura de archivos:**")
st.image("Resources/Images/API_structure.png", use_container_width=False)

st.markdown("**Una vez tenidos todos los archivos docker y la estructura correcta en el proyecto simplemente se realizo una clonación del repositorio con la url especificada en huggingface y se subio el ultimo commit a hugging face para que automaticamente se empezara a buildear el dockerfile**")
# endregion

# Sección: Visualización con Plotly
st.header("Matriz de Confusión con Plotly")
st.markdown("""
Para mejorar la visualización de la matriz de confusión, se utiliza **Plotly**. Esta biblioteca permite crear gráficos 
interactivos y visualmente atractivos. Con Plotly se puede:
- Representar la matriz de confusión de forma dinámica, permitiendo al usuario interactuar y explorar los datos.
- Personalizar los colores y el diseño del gráfico, facilitando la identificación de patrones y errores en las predicciones.
""")

# Sección: Ejemplo de Visualización de la Matriz de Confusión
st.header("Métricas y Visualización de Resultados")
st.markdown("""
Para evaluar el modelo, se generan dos métricas clave: la **matriz de confusión** y el **reporte de clasificación**.
Estas métricas permiten analizar el desempeño del modelo en la clasificación de las diferentes clases y evaluar su precisión,
recall y f1-score. 
""")

# Cargar y visualizar la matriz de confusión (si el archivo existe)
try:
    df_cm = pd.read_csv("Resources/confusion_matrix.csv", index_col=0)
    st.subheader("Matriz de Confusión")
    fig = px.imshow(df_cm, 
                    text_auto=True, 
                    color_continuous_scale="Oranges", 
                    labels=dict(x = "Predicciones",y="Etiquetas Verdaderas"),
                    x=df_cm.columns, 
                    y=df_cm.index,
                    width=600,
                    height=700
                    ) 
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title_standoff=20)
    fig.update_xaxes(title_standoff=20) 
    fig.update_layout(margin=dict(l=50, r=50, t=50, b=150))
    st.plotly_chart(fig, use_container_width=True, theme=None)
except Exception as e:
    st.info("No se encontró el archivo 'confusion_matrix.csv'.")

# Cargar las métricas
try:
    metrics = pd.read_csv("Resources/classification_report.csv", index_col=0)
    st.subheader("Reporte de Clasificación")
    # round the values to 3 decimal points
    metrics = metrics.round(3)    
    st.dataframe(metrics)
except Exception as e:
    st.info("No se encontró el archivo 'classification_report.json'.")

st.markdown("---")
st.markdown("© 2025 UdeC - Documentación para Clasificación de Enfermedades en Hojas de Tomate")
