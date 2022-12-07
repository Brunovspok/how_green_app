from fastapi import FastAPI

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'hi': True}


@app.get('/predict')
def predict():
    return {'pred': True}
