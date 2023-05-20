from PIL import Image
import io
from transformers import pipeline

def classificate(contents):
	image = Image.open(io.BytesIO(contents))
    
	# 1. Descargar el modelo
	model = pipeline("image-classification", model = "microsoft/resnet-50")

	# 2. Pasar la imagen
	result = model(image)

	# 3. Regresar los resultados! 🤯
	return result[0]['label']
	