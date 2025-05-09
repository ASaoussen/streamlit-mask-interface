import streamlit as st
import requests
from PIL import Image
import io
import os

# ---------------- CONFIG ----------------
API_URL = "https://segmentationimages.azurewebsites.net/predict_mask"
IMAGE_FOLDER = "images"  # Dossier contenant les images réelles
MASK_GT_FOLDER = "masks_gt"  # Dossier contenant les masques ground truth

# ---------------- INTERFACE ----------------
st.title("Interface de test de segmentation")
st.markdown("""
Cette application permet de :
- Lister les IDs d'images disponibles,
- Envoyer une image à l’API pour obtenir le masque prédit,
- Afficher l’image réelle, le masque réel, et le masque prédit.
""")

# ---------------- LISTE DES IMAGES ----------------
# Liste des noms de fichiers sans extension
image_ids = [
    os.path.splitext(f)[0]
    for f in os.listdir(IMAGE_FOLDER)
    if f.endswith((".png", ".jpg", ".jpeg"))
]

if not image_ids:
    st.warning("Aucune image trouvée dans le dossier.")
else:
    selected_id = st.selectbox("Choisis un ID d'image :", sorted(image_ids))

    if st.button("Lancer la prédiction") and selected_id:
        try:
            # Charger les fichiers locaux
            image_path = os.path.join(IMAGE_FOLDER, selected_id + ".png")
            mask_gt_path = os.path.join(MASK_GT_FOLDER, selected_id + ".png")

            if not os.path.exists(image_path):
                st.error("Image non trouvée.")
            elif not os.path.exists(mask_gt_path):
                st.error("Masque ground truth non trouvé.")
            else:
                image = Image.open(image_path).convert("RGB")
                mask_gt = Image.open(mask_gt_path)

                # Affichage image et mask ground truth
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Image réelle", use_container_width=True)
                with col2:
                    st.image(mask_gt, caption="Masque réel", use_container_width=True)

                # Envoi à l'API
                with st.spinner("Prédiction en cours..."):
                    with open(image_path, "rb") as f:
                        files = {"image": (f"{selected_id}.png", f, "image/png")}
                        response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        try:
                            pred_mask = Image.open(io.BytesIO(response.content))
                            st.success("Masque prédit reçu avec succès !")
                            st.image(pred_mask, caption="Masque prédit", use_container_width=True)
                        except Exception as e:
                            st.error(f"Erreur de traitement du masque prédit : {e}")
                    else:
                        st.error(f"Erreur API ({response.status_code}) : {response.text}")

        except Exception as e:
            st.error(f"Erreur : {e}")
