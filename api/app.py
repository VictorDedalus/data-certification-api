
import pandas as pd
import joblib
from fastapi import FastAPI

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict/")
def create(acousticness,
			danceability,
           	duration_ms,
            energy,
            explicit,
            id,
            instrumentalness,
            key,
            liveness,
            loudness,
            mode,
            name,
            release_date,
            speechiness,
            tempo,
            valence,
            artist):
        X = pd.DataFrame(dict(
            acousticness = [float(acousticness)],
            danceability = [float(danceability)],
            duration_ms = [int(duration_ms)],
            energy = [float(energy)],
            explicit = [int(explicit)],
            id = [id],
            instrumentalness = [float(instrumentalness)],
            key = [int(key)],
            liveness = [float(liveness)],
            loudness = [float(loudness)],
            mode = [int(mode)],
            name = [name],
            release_date = [release_date],
            speechiness = [float(speechiness)],
            tempo = [float(tempo)],
            valence = [float(valence)],
            artist = [artist]))
            
        pipeline = joblib.load('model.joblib')

        results = pipeline.predict(X)

        pred = float(results[0])

        return dict(artist=artist,
        name=name,
        popularity=pred)
            				