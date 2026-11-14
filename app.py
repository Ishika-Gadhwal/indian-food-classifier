import json
import torch
import torch.nn as nn
import gradio as gr
from torchvision import models, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
class_names = json.load(open("class_names.json"))

model = models.efficientnet_b0(weights=None)
in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, len(class_names)))
model.load_state_dict(torch.load("food_classifier.pt", map_location=device))
model.to(device).eval()

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(img):
    if img is None:
        return {}
    x = eval_tf(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0]
    top = probs.topk(3)
    result = {class_names[i]: float(p) for p, i in zip(top.values, top.indices)}
    if top.values[0] < 0.30:
        result = {"not sure (maybe not in my 80 dishes)": 1.0}
    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a food photo"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 predictions"),
    title="Indian Food Classifier",
)

if __name__ == "__main__":
    demo.launch()