from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

import pandas as pd
import joblib

from tensorflow import keras


app = FastAPI()

templates = Jinja2Templates(directory="templates")


# Load model and scaler
model = keras.models.load_model("ann_model.keras")
scaler = joblib.load("scaler.pkl")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(
    request: Request,

    MedInc: float = Form(...),
    HouseAge: float = Form(...),
    AveRooms: float = Form(...),
    AveBedrms: float = Form(...),
    Population: float = Form(...),
    AveOccup: float = Form(...),
    Latitude: float = Form(...),
    Longitude: float = Form(...)
):

    data = pd.DataFrame([[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]], columns=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ])

    # Scale input
    data_scaled = scaler.transform(data)

    # ANN prediction
    prediction = model.predict(data_scaled, verbose=0)[0][0]

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "prediction": round(float(prediction), 4)
        }
    )
