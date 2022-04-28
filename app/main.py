from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from utils import model64

app = FastAPI()

@app.get('/')
def index():
    return  {
        'Project Name': 'Melanoma Detection',
        'Subject': 'Fundamentals of Aritificial Intelligence',
        'Course Code': 'CSE2039',
        'Slot': 'F1',
        'Members': {
            'Kavya Palliwal': '20BRS1111',
            'Hritik Goel': '20BRS1035',
            'Aniket Kumar Paul': '20BRS1116',
            'G Aneesh Aparajit': '20BRS1054'
        },
        'Faculty': 'Dr. R Jothi',
        'Next Step': "Go to '/docs' to test the model. You can test the model at the '/predict' endpoint."
    }
    
pred_dict = {0: 'benign', 1: 'malignant'}
    
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((64, 64))
    image = np.asarray(image)
    return image

MODEL_RESNET = tf.keras.models.load_model('../../checkpoints/resnet')

print(MODEL_RESNET.summary())

@app.post('/resnet')
async def index(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image = np.expand_dims(image, 0)
    predictions = MODEL_RESNET.predict(image)
    idx = np.argmax(predictions[0]).tolist()
    print(predictions[0], idx)
    label = pred_dict[idx]
    return {
        'data': {
            'prediction': label,
            'idx': idx,
            'softmax': predictions[0].tolist()
        }
    }
    
MODEL_64 = tf.keras.models.load_model('../../checkpoints/model64')
@app.post('/model64')
async def index(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image = np.expand_dims(image, 0)
    predictions = MODEL_64.predict(image)
    idx = np.argmax(predictions[0]).tolist()
    print(predictions[0], idx)
    label = pred_dict[idx]
    return {
        'data': {
            'prediction': label,
            'idx': idx,
            'softmax': predictions[0].tolist()
        }
    }

if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=8080)