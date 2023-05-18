from PIL import Image
import io
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch

def classificate(content):
	image = Image.open(io.BytesIO(content))
	processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
	model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

	inputs = processor(image, return_tensors="pt")

	with torch.no_grad():
		logits = model(**inputs).logits

	# model predicts one of the 1000 ImageNet classes
	predicted_label = logits.argmax(-1).item()

	return model.config.id2label[predicted_label]