import streamlit as st
import requests
from PIL import Image
import io


# ---------------- CONFIG ----------------
API_URL = "https://segmentationimages.azurewebsites.net/predict_mask"  # Ajout du https obligatoire

# ---------------- INTERFACE ----------------
st.title("Segmentation d'image")
st.markdown("Charge une image, envoie-la à l'API Flask et récupère le masque prédit.")

uploaded_file = st.file_uploader("Choisis une image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Affiche l'image originale
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image originale", use_container_width=True)

        # Préparation des données pour l'API
        files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        # Appel API
        with st.spinner("Prédiction en cours..."):
            response = requests.post(API_URL, files=files)

        # Vérification du statut de la réponse
        print(response.status_code, response.content)
        if response.status_code == 200:
            try:
                mask_image = Image.open(io.BytesIO(response.content))
                st.success("Masque généré avec succès !")
                st.image(mask_image, caption="Masque segmenté", use_container_width=True)
            except Exception as e:
                st.error(f"Erreur lors du traitement du masque : {e}")
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")

    except Exception as e:
        st.error(f"Erreur de traitement : {e}")
