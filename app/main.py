from fastapi import FastAPI, UploadFile, File
import uvicorn

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
    
@app.post('/predict')
def index(file: UploadFile = File(...)):
    pass

if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=8080)