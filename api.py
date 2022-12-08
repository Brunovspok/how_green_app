from fastapi import FastAPI, UploadFile
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'hi': True}

@app.post("/predict/")
async def create_upload_file(file: UploadFile):

    content = await file.read()
    X_train = np.frombuffer(content, dtype=np.float32)
    X_train = X_train.reshape(int(len(X_train)/3), 3)
    model = load_model('my_model')
    dataset_test = timeseries_dataset_from_array(
        X_train,
        np.ones(48),
        sequence_length=50,
        batch_size=32,
    )
    pred = model.predict(dataset_test)
    return {'pred': np.array(pred).tolist()}
