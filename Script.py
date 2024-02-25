import numpy as np
from tensorflow import keras
from PIL import Image

# Cargar el modelo desde el archivo h5
model = keras.models.load_model("perros-gatos-cnn-ad.h5")
# Cargar la imagen desde el archivo jpg
img = Image.open("test.jpg")
# Mostrar imagen
img.show()
# Convertir la imagen a escala de grises
img = img.convert("L")
# Redimensionar la imagen a 100x100
img = img.resize((100, 100))
# Convertir la imagen a un array de numpy
img = np.array(img)
# Añadir una dimensión extra para indicar que es una sola imagen
img = img[np.newaxis, :, :, np.newaxis]
# Normalizar los valores de los píxeles entre 0 y 1
img = img / 255.0
# Hacer inferencia con el modelo
pred = model.predict(img)
# Mostrar los resultados
print(pred)
if pred[0][0] > 0.5:
    print("Perro")
else:
    print("Gato")