import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

api_meterologia = "https://single-runs-api.open-meteo.com/v1/forecast"

def pegar_dados(local, data_inicio, data_fim):
    
    # Pégando os dados metoloógicos
    params_meterologia = {
        "latitude": local[0],
        "longitude": local[1],
        "start_date": data_inicio,
        "end_date": data_fim,
        "daily": "temperature_2m",
        "models": "ecmwf_ifs",
    }

    dados_meterologicos = openmeteo_requests.get(api_meterologia, params=params_meterologia)

    df = pd.DataFrame(dados_meterologicos["daily"])
    print(df.head())
    
def preprocessar_dados(df):
    return df
    
def construir_modelo():
    modelo = Sequential()
    modelo.add(SimpleRNN(32, input_shape=(10, 3), activation='relu'))
    modelo.add(Dense(1))
    
    modelo.compile(optimizer='adam', loss='mse')
    
    return modelo

if __name__ == "__main__":
    localização = (-10.184, -48.333)
    data_inicio = "2023-01-01"
    data_fim = "2023-01-31"
    parametros_analisados =  [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m"
    ]
    
    pegar_dados(localização, data_inicio, data_fim, parametros_analisados)
    