"""Fashion Item Classifier - a Streamlit app built on Hager's Fashion-MNIST CNN."""
import os
import glob
import re
import torch
import streamlit as st
from PIL import Image, ImageOps

from model import CNN_batch, CLASS_NAMES

st.set_page_config(page_title="What am I wearing?", page_icon="👕", layout="centered")

@st.cache_resource
def load_model():
    model = CNN_batch(out_1=16, out_2=32, number_of_classes=10)
    model.load_state_dict(torch.load(glob.glob("*fashion_mnist_cnn.pth")[0], map_location="cpu", weights_only=True))
    model.eval()
    return model

def preprocess(img):
    """Turn any image into the 28x28 grayscale tensor the model expects."""
    img = img.convert("L")                      # grayscale
    img = ImageOps.fit(img, (28, 28))           # resize to 28x28
    # Fashion-MNIST images are light clothing on a dark background.
    # If the uploaded photo is the opposite (dark item on light background), invert it.
    if sum(img.getdata()) / (28 * 28) > 127:
        img = ImageOps.invert(img)
    import torchvision.transforms as transforms
    return transforms.ToTensor()(img).unsqueeze(0)  # shape: 1 x 1 x 28 x 28

def predict(model, img):
    tensor = preprocess(img)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    return probs

# ---------- UI ----------
st.title("👗 What am I wearing?")
st.write(
    "Upload a photo of a clothing item and a small neural network "
    "(a CNN trained on the Fashion-MNIST dataset) will guess what it is. "
    "It knows 10 categories: " + ", ".join(CLASS_NAMES) + "."
)

model = load_model()

tab_upload, tab_sample = st.tabs(["📤 Upload your own image", "🎲 Try a sample image"])

img = None
with tab_upload:
    uploaded = st.file_uploader("Choose an image (photo of a shirt, shoe, bag...)", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if uploaded:
        img = Image.open(uploaded)

with tab_sample:
    sample_files = sorted(os.listdir("samples")) if os.path.isdir("samples") else []
    cols = st.columns(5)
    for i, fname in enumerate(sample_files):
        with cols[i % 5]:
            st.image(f"samples/{fname}", width=60)
            if st.button(re.sub(r"^\d+-", "", fname).replace(".png", ""), key=fname):
                img = Image.open(f"samples/{fname}")

if img is not None:
    probs = predict(model, img)
    top_prob, top_idx = torch.max(probs, 0)

    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img.resize((140, 140)), caption="your image (resized)")
    with col2:
        st.subheader(f"Prediction: {CLASS_NAMES[top_idx]} 🎉")
        st.write(f"Confidence: **{top_prob.item()*100:.1f}%**")

    st.write("How sure the model is about each category:")
    chart_data = {CLASS_NAMES[i]: probs[i].item() for i in range(len(CLASS_NAMES))}
    st.bar_chart(chart_data, horizontal=True)

    st.caption(
        "Tip: the model was trained on tiny 28x28 grayscale images, so clear photos "
        "of a single item work best. It can struggle with tricky photos - that is normal!"
    )

st.divider()
st.caption("Built by Hager Abd El Hakim · CNN in PyTorch · trained on Fashion-MNIST")
