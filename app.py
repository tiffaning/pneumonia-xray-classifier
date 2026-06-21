import torch
import gradio as gr
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# ── device ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── rebuild model architecture ───────────────────────────
def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(DEVICE)

model = build_model()

# ── load saved weights ───────────────────────────────────
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()

# ── image transform ──────────────────────────────────────
tf_eval = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── prediction function ──────────────────────────────────
@torch.no_grad()
def predict_image(image):
    x = tf_eval(image).unsqueeze(0).to(DEVICE)
    p = torch.softmax(model(x), 1)[0].cpu().numpy()
    return {"NORMAL": float(p[0]), "PNEUMONIA": float(p[1])}

# ── gradio interface ─────────────────────────────────────
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Pneumonia X-Ray Classifier",
    description="Upload a chest X-ray image. The model will predict Normal vs Pneumonia.",
)

demo.launch()