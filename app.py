import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import os

class AdversarialAttacks:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def fgsm_attack(self, images, labels, eps=0.02):
        images = images.clone().detach()
        images.requires_grad = True
        
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        grad_sign = images.grad.data.sign()
        perturbed_images = images + eps * grad_sign
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()

    def pgd_attack(self, images, labels, eps=0.02, alpha=0.01, iters=10):
        perturbed_images = images.clone().detach()
        perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-eps, eps)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        for i in range(iters):
            perturbed_images.requires_grad = True
            outputs = self.model(perturbed_images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            with torch.no_grad():
                grad_sign = perturbed_images.grad.data.sign()
                perturbed_images = perturbed_images + alpha * grad_sign
                eta = torch.clamp(perturbed_images - images, -eps, eps)
                perturbed_images = torch.clamp(images + eta, 0, 1)
        
        return perturbed_images.detach()

    def bim_attack(self, images, labels, eps=0.02, alpha=0.01, iters=10):
        perturbed_images = images.clone().detach()
        
        for i in range(iters):
            perturbed_images.requires_grad = True
            outputs = self.model(perturbed_images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            with torch.no_grad():
                grad_sign = perturbed_images.grad.data.sign()
                perturbed_images = perturbed_images + alpha * grad_sign
                perturbed_images = torch.clamp(perturbed_images, images - eps, images + eps)
                perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()

def load_model(model_path, device):
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.aux_logits = False
    model.fc = nn.Linear(model.fc.in_features, 15)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def process_image(image, device):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def get_prediction(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = F.softmax(outputs, dim=1)
    return class_names[predicted.item()], probabilities[0]

def plot_perturbation_heatmap(original, perturbed):
    perturbation = (perturbed - original).abs()
    perturbation_map = perturbation[0].cpu().mean(dim=0).numpy()
    fig = px.imshow(perturbation_map, 
                    color_continuous_scale='Viridis',
                    title='Perturbation Heatmap')
    return fig

def plot_prediction_probabilities(probabilities, class_names):
    probs = probabilities.cpu().detach().numpy()
    fig = go.Figure(data=[
        go.Bar(x=class_names, y=probs, 
               marker_color='rgb(55, 83, 109)')
    ])
    fig.update_layout(
        title='Class Prediction Probabilities',
        xaxis_title='Traffic Sign Classes',
        yaxis_title='Probability',
        xaxis_tickangle=-45
    )
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Adversarial Attack Demo")
    
    # Title and Introduction
    st.title("🛡️ Traffic Sign Recognition: Adversarial Attack Analysis")
    
    st.markdown("""
    ## Project Overview
    This application demonstrates the vulnerability of deep learning models to adversarial attacks in the context of traffic sign recognition. 
    We explore three popular attack methods:
    
    1. **FGSM (Fast Gradient Sign Method)**: A one-step attack that perturbs images based on the sign of the gradient
    2. **PGD (Projected Gradient Descent)**: An iterative attack that creates stronger adversarial examples
    3. **BIM (Basic Iterative Method)**: A refined version of FGSM that applies smaller changes over multiple iterations
    
    ### Academic Significance
    This project investigates the robustness of deep learning models in safety-critical applications, specifically:
    - Model vulnerability to imperceptible perturbations
    - Implications for autonomous driving systems
    - Comparison of different attack methodologies
    - Evaluation of defensive strategies
    """)

    # Load model and setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = load_model('best_model.pth', device)
        adv_model = load_model('adversarial_model.pth', device)
    except Exception as e:
        st.error(f"Error: Model files not found. Please ensure 'best_model.pth' and 'adversarial_model.pth' exist. {str(e)}")
        return

    attacks = AdversarialAttacks(model, device)
    
    class_names = [
        'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 20',
        'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60',
        'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Speed Limit 100',
        'Speed Limit 110', 'Speed Limit 120', 'Stop'
    ]

    # Sidebar controls
    st.sidebar.title("Control Panel")
    
    # Image selection
    image_source = st.sidebar.radio("Select Image Source", ["Upload Image", "Use Sample Images"])
    
    image = None
    if image_source == "Use Sample Images":
        sample_image_dir = "sample_images"
        try:
            sample_images = [f for f in os.listdir(sample_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if sample_images:
                selected_sample = st.sidebar.selectbox("Select a sample image", sample_images)
                image_path = os.path.join(sample_image_dir, selected_sample)
                image = Image.open(image_path).convert('RGB')
            else:
                st.sidebar.error("No sample images found in the sample_images directory")
        except FileNotFoundError:
            st.sidebar.error("Sample images directory not found. Please create a 'sample_images' directory.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Traffic Sign Image", type=["jpg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    
    attack_type = st.sidebar.selectbox(
        "Select Attack Method",
        ["FGSM", "PGD", "BIM"]
    )
    
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Standard Model", "Adversarially Trained Model"]
    )
    
    eps = st.sidebar.slider("Perturbation Magnitude (ε)", 0.0, 0.1, 0.02, 0.01)
    
    if attack_type in ["PGD", "BIM"]:
        alpha = st.sidebar.slider("Step Size (α)", 0.001, 0.05, 0.01, 0.001)
        iters = st.sidebar.slider("Number of Iterations", 1, 20, 10)

    # Main content
    if image is not None:
        # Process image
        image_tensor = process_image(image, device)
        
        # Select model
        active_model = adv_model if model_type == "Adversarially Trained Model" else model
        
        # Get original prediction
        orig_pred, orig_probs = get_prediction(active_model, image_tensor, class_names)
        
        # Generate adversarial example
        label = torch.tensor([class_names.index(orig_pred)]).to(device)
        
        if attack_type == "FGSM":
            perturbed_tensor = attacks.fgsm_attack(image_tensor, label, eps=eps)
        elif attack_type == "PGD":
            perturbed_tensor = attacks.pgd_attack(image_tensor, label, eps=eps, alpha=alpha, iters=iters)
        else:  # BIM
            perturbed_tensor = attacks.bim_attack(image_tensor, label, eps=eps, alpha=alpha, iters=iters)
        
        # Get adversarial prediction
        adv_pred, adv_probs = get_prediction(active_model, perturbed_tensor, class_names)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            st.write(f"Prediction: {orig_pred}")
            st.plotly_chart(plot_prediction_probabilities(orig_probs, class_names))
        
        with col2:
            st.subheader("Adversarial Image")
            perturbed_image = transforms.ToPILImage()(perturbed_tensor.squeeze().cpu())
            st.image(perturbed_image, use_container_width=True)
            st.write(f"Prediction: {adv_pred}")
            st.plotly_chart(plot_prediction_probabilities(adv_probs, class_names))
        
        # Show perturbation heatmap
        st.subheader("Attack Analysis")
        st.plotly_chart(plot_perturbation_heatmap(image_tensor, perturbed_tensor))
        
        # Attack success metrics
        success = orig_pred != adv_pred
        st.write(f"""
        ### Attack Results
        - **Attack Success**: {"✅ Yes" if success else "❌ No"}
        - **Original Prediction**: {orig_pred} ({orig_probs.max().item():.2%} confidence)
        - **Adversarial Prediction**: {adv_pred} ({adv_probs.max().item():.2%} confidence)
        - **Perturbation Magnitude**: {eps:.3f}
        """)
        
        # Technical details
        with st.expander("Technical Details"):
            st.write(f"""
            ### Model Architecture
            - Base Model: Inception V3
            - Modified for 15 traffic sign classes
            - {"Adversarially trained" if model_type == "Adversarially Trained Model" else "Standard training"}
            
            ### Attack Parameters
            - Method: {attack_type}
            - Epsilon (ε): {eps}
            {"- Alpha (α): " + str(alpha) if attack_type in ["PGD", "BIM"] else ""}
            {"- Iterations: " + str(iters) if attack_type in ["PGD", "BIM"] else ""}
            """)
    
    else:
        st.info("Please select or upload a traffic sign image to begin the analysis.")

if __name__ == "__main__":
    main()