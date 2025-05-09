import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
import io

# ---------------- CONFIG ----------------
API_URL = "https://segmentationimages.azurewebsites.net/predict_mask"  # URL de l'API sur Azure

# ---------------- INTERFACE ----------------
st.title("Interface de test de segmentation")
st.markdown("""
Cette application permet de :
- Lister les IDs d'images disponibles,
- Envoyer une image à l’API pour obtenir le masque prédit,
- Afficher l’image réelle, le masque réel, et le masque prédit.
""")

# ---------------- CHARGEMENT DE L'IMAGE ----------------
uploaded_file = st.file_uploader("Choisis une image", type=["png", "jpg", "jpeg"])

# ---------------- CHARGEMENT DU MASQUE RÉEL ----------------
# Tu pourrais ici choisir un chemin ou un mécanisme pour associer l'image téléchargée avec son masque réel
mask_real_file = None
if uploaded_file is not None:
    try:
        # Affiche l'image téléchargée
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image originale", use_container_width=True)

        # Remplace cette ligne par ton mécanisme pour récupérer le masque réel correspondant
        # (par exemple, tu pourrais avoir un dossier de masques réels avec un nom d'image similaire)
        mask_real_file = uploaded_file.name.replace(".jpg", "_mask.png")  # Exemple, tu peux personnaliser cette logique
        try:
            mask_real = Image.open(mask_real_file)
            st.image(mask_real, caption="Masque réel", use_container_width=True)
        except FileNotFoundError:
            st.warning("Le masque réel pour cette image n'a pas été trouvé.")

        # ---------------- ENVOI À L'API ----------------
        with st.spinner("Prédiction en cours..."):
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(API_URL, files=files)

        # Vérification de la réponse de l'API
        if response.status_code == 200:
            try:
                # Essaye d'ouvrir le masque généré par l'API
                pred_mask = Image.open(io.BytesIO(response.content))
                st.success("Masque prédit reçu avec succès !")
                st.image(pred_mask, caption="Masque prédit", use_container_width=True)
            except UnidentifiedImageError:
                st.error("Le fichier renvoyé par l'API n'est pas une image valide.")
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")

    except Exception as e:
        st.error(f"Erreur de traitement : {e}")
