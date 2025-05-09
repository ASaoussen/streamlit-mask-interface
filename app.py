import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
import io
import os

# ---------------- CONFIG ----------------
API_URL = "https://segmentationimages.azurewebsites.net/predict_mask"  # URL de l'API sur Azure
BASE_IMAGE_FOLDER = "./data/leftImg8bit/val"  # Chemin relatif vers le dossier des images
BASE_MASK_FOLDER = "./data/gtFine/val"  # Chemin relatif vers le dossier des masques réels

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
if uploaded_file is not None:
    try:
        # Affiche l'image téléchargée
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image originale", use_container_width=True)

        # Utiliser l'ID de l'image pour trouver le masque réel dans le dossier correspondant
        img_filename = uploaded_file.name
        selected_id = img_filename.replace("_leftImg8bit.png", "")
        mask_filename = selected_id + "_gtFine_color.png"

        # Extraire le nom de la ville (par ex. 'frankfurt')
        city_name = selected_id.split("_")[0]

        # Définir le chemin vers le masque réel
        mask_real_path = os.path.join(BASE_MASK_FOLDER, city_name, mask_filename)

        # Log du chemin du masque réel pour débogage
        st.write(f"Chemin du masque réel : {mask_real_path}")

        # Vérifie si le masque réel existe dans le répertoire
        if os.path.exists(mask_real_path):
            mask_real = Image.open(mask_real_path)
            st.image(mask_real, caption="Masque réel", use_container_width=True)
        else:
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
