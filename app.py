from pydantic import BaseModel, Field
import pickle
from fastapi import FastAPI

app = FastAPI()


def load_model():
    with open('classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


class WineFeatures(BaseModel):
    alcohol: float = Field(..., example=13.2)
    malic_acid: float = Field(..., example=1.78)
    ash: float = Field(..., example=2.14)
    alcalinity_of_ash: float = Field(..., example=11.2)
    magnesium: float = Field(..., example=100.0)
    total_phenols: float = Field(..., example=2.65)
    flavanoids: float = Field(..., example=2.76)
    nonflavanoid_phenols: float = Field(..., example=0.26)
    proanthocyanins: float = Field(..., example=1.28)
    color_intensity: float = Field(..., example=4.38)
    hue: float = Field(..., example=1.05)
    od280_od315_of_diluted_wines: float = Field(..., example=3.4)
    proline: float = Field(..., example=1050.0)


@app.post("/predict")
def predict_wine(features: WineFeatures):
    model = load_model()
    input_data = [[
        features.alcohol, features.malic_acid, features.ash, features.alcalinity_of_ash,
        features.magnesium, features.total_phenols, features.flavanoids,
        features.nonflavanoid_phenols, features.proanthocyanins, features.color_intensity,
        features.hue, features.od280_od315_of_diluted_wines, features.proline
    ]]

    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
