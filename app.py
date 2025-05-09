import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
import io
from urllib.request import urlopen

# ---------------- CONFIG ----------------
API_URL = "https://segmentationimages.azurewebsites.net/predict_mask"  # URL de l'API sur Azure
GITHUB_MASK_BASE_URL = "https://raw.githubusercontent.com/ASaoussen/streamlit-mask-interface/master/data/gtFine/val"

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

        # Utiliser l'ID de l'image pour retrouver le masque
        img_filename = uploaded_file.name
        selected_id = img_filename.replace("_leftImg8bit.png", "").replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        mask_filename = selected_id + "_gtFine_color.png"

        # Extraire le nom de la ville (par ex. 'frankfurt')
        city_name = selected_id.split("_")[0]

        # Construire l'URL du masque réel sur GitHub
        mask_url = f"{GITHUB_MASK_BASE_URL}/{city_name}/{mask_filename}"
        st.write(f"URL du masque réel : {mask_url}")

        # Charger le masque réel depuis GitHub
        try:
            with urlopen(mask_url) as response:
                mask_real = Image.open(response)
                st.image(mask_real, caption="Masque réel (depuis GitHub)", use_container_width=True)
        except Exception as e:
            st.warning(f"Le masque réel n’a pas été trouvé sur GitHub. Détails : {e}")

        # ---------------- ENVOI À L'API ----------------
        with st.spinner("Prédiction en cours..."):
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(API_URL, files=files)

        # Vérification de la réponse de l'API
        if response.status_code == 200:
            try:
                pred_mask = Image.open(io.BytesIO(response.content))
                st.success("Masque prédit reçu avec succès !")
                st.image(pred_mask, caption="Masque prédit", use_container_width=True)
            except UnidentifiedImageError:
                st.error("Le fichier renvoyé par l'API n'est pas une image valide.")
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")

    except Exception as e:
        st.error(f"Erreur de traitement : {e}")
