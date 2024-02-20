from flask import Flask, render_template
from flask import request
from PIL import Image
import io
import base64
import cv2
import Digit_Recognizer 
from Digit_Recognizer import process_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    json_data = request.get_json()  # Récupérer le JSON de la requête POST
    image_data = json_data['image']  # Récupérer les données de l'image base64 depuis le JSON

    # Supprimer le préfixe "data:image/jpeg;base64," du texte base64
    base64_data = image_data.split(',')[1]

    # Décoder les données base64 en bytes
    image_bytes = base64.b64decode(base64_data)

    # Ouvrir l'image à partir des bytes
    image = Image.open(io.BytesIO(image_bytes))

    # Afficher l'image
    image.show()
    # Appeler la fonction pour traiter l'image
    

    return 'Traitement du téléchargement...'

if __name__ == '__main__':
    app.run()