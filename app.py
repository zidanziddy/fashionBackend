from flask import Flask, request, jsonify
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import io

app = Flask(__name__)

# Load the model and processor
model_path = 'skin_types_image_detection'
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
model.eval()

def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    id2label = model.config.id2label
    predicted_label = id2label[predicted_class_idx]
    return predicted_label

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img_bytes = file.read()
        prediction = predict(img_bytes)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
