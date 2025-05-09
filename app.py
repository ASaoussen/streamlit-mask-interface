import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
import io
import os
import glob

# ---------------- CONFIG ----------------
API_URL = "https://segmentationimages.azurewebsites.net/predict_mask"
BASE_IMAGE_FOLDER = "C:\\Users\\attia\\data\\leftImg8bit\\val"
BASE_MASK_FOLDER =  "C:\\Users\\attia\\data\\gtFine\\val"

# ---------------- INTERFACE ----------------
st.title("Interface de test de segmentation")
st.markdown("""
Cette application permet de :
- Lister les IDs d'images disponibles,
- Envoyer une image à l’API pour obtenir le masque prédit,
- Afficher l’image réelle, le masque réel, et le masque prédit.
""")

# ---------------- LISTE DES IMAGES ----------------
# Utilisation de glob pour parcourir les sous-dossiers des différentes villes (directories)
image_paths = glob.glob(os.path.join(BASE_IMAGE_FOLDER, "**", "*_leftImg8bit.png"), recursive=True)
image_ids = [
    os.path.basename(p).replace("_leftImg8bit.png", "")
    for p in image_paths
]

if not image_ids:
    st.warning("Aucune image trouvée dans le dossier.")
else:
    selected_id = st.selectbox("Choisis un ID d'image :", sorted(image_ids))

    if st.button("Lancer la prédiction") and selected_id:
        try:
            # Vérifie la validité de l'ID sélectionné
            parts = selected_id.split("_")
            if len(parts) < 3:
                st.error("Format d'ID invalide. Exemple attendu : frankfurt_000000_000294")
            else:
                city = parts[0]
                img_filename = selected_id + "_leftImg8bit.png"
                mask_filename = selected_id + "_gtFine_color.png"

                # Construction des chemins absolus
                image_path = os.path.join(BASE_IMAGE_FOLDER, city, img_filename)
                mask_gt_path = os.path.join(BASE_MASK_FOLDER, city, mask_filename)

                if not os.path.exists(image_path):
                    st.error(f"Image non trouvée : {image_path}")
                elif not os.path.exists(mask_gt_path):
                    st.error(f"Masque ground truth non trouvé : {mask_gt_path}")
                else:
                    image = Image.open(image_path).convert("RGB")
                    mask_gt = Image.open(mask_gt_path)

                    # Affichage des images réelles
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Image réelle")
                    with col2:
                        st.image(mask_gt, caption="Masque réel")

                    # Envoi de l'image à l'API
                    with st.spinner("Prédiction en cours..."):
                        with open(image_path, "rb") as f:
                            files = {"image": (img_filename, f, "image/png")}
                            response = requests.post(API_URL, files=files)

                        if response.status_code == 200:
                            try:
                                pred_mask = Image.open(io.BytesIO(response.content))
                                st.success("Masque prédit reçu avec succès !")
                                st.image(pred_mask, caption="Masque prédit")
                            except UnidentifiedImageError:
                                st.error("Le fichier renvoyé par l'API n'est pas une image valide.")
                        else:
                            st.error(f"Erreur API ({response.status_code}) : {response.text}")

        except Exception as e:
            st.error(f"Erreur : {e}")
