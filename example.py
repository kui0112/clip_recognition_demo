import torch
import clip
from PIL import Image

# 加载模型
if not torch.cuda.is_available():
    raise Exception("cuda environment error.")
device = "cuda"
model, preprocess = clip.load("ViT-B/32", device=device, download_root=r"D:\models")
image = preprocess(Image.open("files/apple.jpg")).unsqueeze(0).to(device)
labels = ["a pear", "an apple", "a peach", "a dog", "a cat"]
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

probs = probs[0]
for i, label in enumerate(labels):
    print(f"{label}: {probs[i]}")
