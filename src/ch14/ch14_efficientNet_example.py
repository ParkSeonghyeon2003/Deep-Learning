import torch
from datasets import load_dataset
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

dataset = load_dataset("huggingface/cats-image", 
                       trust_remote_code=True)
image = dataset["test"]["image"][0]
image.show(image)

preprocessor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b0")
model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0")

inputs = preprocessor(image, return_tensors="pt")

# predict
with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
