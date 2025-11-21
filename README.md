# **ApateShield**  

🚦 *Unmasking the Deception in Deep Learning* 🚦  

Deployed on [apate-shield.streamlit.app](https://apate-shield.streamlit.app)

ApateShield is an innovative project exploring the vulnerabilities of deep learning models in traffic sign recognition systems when faced with adversarial attacks. Leveraging **InceptionNet**, this project delves into the stark difference between predictions under normal and adversarial conditions.  

With a sleek **Streamlit interface**, it’s as much about interaction as it is about education. You’ll not only see how adversarial attacks work but also test a robust adversarially trained model built to resist them.  

---

## **✨ Key Features**  

- **🔍 Model Training**: Fine-tuned InceptionNet for recognizing traffic signs.  
- **⚔️ Adversarial Attacks**: Implementation of FGSM, PGD, and BIM methods.  
- **🛡️ Adversarial Training**: Builds a defense-ready model to counter adversarial attacks.  
- **📊 Interactive Visuals**: Explore and compare results through a Streamlit app.  

---

## **🚀 Getting Started**  

### 1️⃣ Clone the Repository  
```bash  
git clone https://github.com/yourusername/ApateShield.git  
cd ApateShield  
```  

### 2️⃣ Set Up a Virtual Environment *(Optional but Recommended)*  

**On Linux/macOS:**  
```bash  
python3 -m venv venv  
source venv/bin/activate  
```  

**On Windows:**  
```bash  
python -m venv venv  
venv\Scripts\activate  
```  

### 3️⃣ Install Dependencies  
```bash  
pip install -r requirements.txt  
```  

---

## **📚 Usage Guide**  

1. **📂 Download the Dataset**  
   - Get the Traffic Signs Dataset: [Traffic Signs Detection Dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection).  
   - Extract it into the `data` directory.  

2. **🤖 Train Models**  
   - Use the training scripts to build both a standard and adversarially robust model.  
   - Save the models as `best_model.pth` and `adversarial_model.pth` in the project root.  

3. **🌐 Launch the Streamlit App**  
   ```bash  
   streamlit run app.py  
   ```  

4. **🎮 Interact with the App**  
   - Upload traffic sign images or use existing ones.  
   - Experiment with various attack methods.  
   - Visualize predictions and compare results between models.  

---

## **📦 Dependencies**  

- Python 3.7+  
- torch  
- torchvision  
- streamlit  
- numpy  
- matplotlib  
- plotly  
- Pillow  

Install all dependencies at once:  
```bash  
pip install -r requirements.txt  
```  

---

## **🤝 Contributing**  

💡 Got ideas? Found a bug? Contributions are welcome! Feel free to:  
- Open an issue  
- Submit a pull request  

Let’s make ApateShield even better together! 🚀  

---

## **📜 License**  

This project is licensed under the **GNU GPU License**. Check out the [LICENSE](LICENSE) file for details.  

---

## **🌟 Acknowledgments**  

- **Dataset**: Traffic Signs Detection Dataset.  
- **Inspiration**: The Greek goddess Apate, symbolizing deceit and embodying the adversarial attack concept.  

---

## **⚠️ Disclaimer**  

This project is for **educational and research purposes only**. Adversarial attacks can pose serious risks in real-world applications. Exercise caution while deploying models in sensitive environments.  
