from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb, gray2rgb
from joblib import load
import pickle

# fonction pour prédire la catégorie d'une image donnée
def predict_image_category(image_path):
    # lecture de l'image à partir du chemin donné
    img = imread(image_path)
    
    # si l'image est en niveaux de gris, on la convertit en RGB
    if len(img.shape) == 2:
        img = gray2rgb(img)
        
    # si l'image est en RGBA (A = couche alpha pour la transparence), on la convertit en RGB
    elif img.shape[2] == 4:
        img = rgba2rgb(img)

    # on redimensionne l'image
    img_resized = resize(img, (224, 224), mode='reflect')

    # Aplatissement de l'image (la convertir en un vecteur à une dimension)
    img_flattened = img_resized.flatten()

    # On convertit le vecteur en une matrice 2D pour la prédiction du modèle
    img_2d = img_flattened.reshape(1, -1)

    # Charger le modèle de classification binaire pré-entraîné
    with open("skin\\coco_vs_skinlesions\\finalized_model_coco_skinlesions_99.pkl", 'rb') as f:
        loaded_model = pickle.load(f)

    # Faire la prédiction sur l'image fournie
    prediction = loaded_model.predict(img_2d)
    
    # Interpréter la prédiction du modèle
    output = None
    if prediction[0] == 0:
        output = "coco"
    elif prediction[0] == 1:
        output = "skin_lesion"
    
    # Renvoyer la catégorie prédite
    return output
