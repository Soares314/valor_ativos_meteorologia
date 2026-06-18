import openmeteo_requests
import yfinance as yf
from retry_requests import retry
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

api_meterologia = "https://archive-api.open-meteo.com/v1/archive"
openmeteo = openmeteo_requests.Client()

def pegar_dados(local, data_inicio, data_fim, parametros_analisados, ativos):
    
    
    dados_meterologicos = None
    dados_acoes = None
    
    # Pegando os dados metoloógicos
    params_meterologia = {
        "latitude": local[0],
        "longitude": local[1],
        "start_date": data_inicio,
        "end_date": data_fim,
        "daily": ",".join(parametros_analisados),
    }

    responses = openmeteo.weather_api(api_meterologia, params=params_meterologia)
    dados_meterologicos = responses[0].Daily()
    
    print("\nData de início:")
    print(pd.to_datetime(dados_meterologicos.Time(), unit = "s"))
    print("Data de fim:")
    print(pd.to_datetime(dados_meterologicos.TimeEnd(), unit = "s"))
    
    print("\nTipos de responses:")
    print(type(responses))
    
    print("Tipo de response[0]:")
    print(type(responses[0]))
    
    print("Tipo de dados metereológicos:")
    print(type(dados_meterologicos))
    
    """
    dados_meterologicos_format = converter_daily(dados_meterologicos)

    df = pd.DataFrame(dados_meterologicos_format, columns=parametros_analisados)
    print("Dados metereológicos:")
    print(df.head())
    """
    
    # Pegando os dados das ações e já convertendo os tipos de dados para float32
    dados_acoes = yf.download(ativos, start=data_inicio, end=data_fim)
    
    cols = dados_acoes.select_dtypes(include=["float64", "int64"]).columns
    dados_acoes[cols] = dados_acoes[cols].astype("float32")
    
    return {
            "meterologia": dados_meterologicos, 
            "ações": dados_acoes
           }    
    
def preprocessar_dados(dados_meterologicos, parametros_analisados):
    
    # Passando os dados para um dicionario e depois convertendo os valores para float32
    dados_meterologicos_format = { "date": pd.date_range(
        start = pd.to_datetime(dados_meterologicos.Time(), unit = "s"),
        end = pd.to_datetime(dados_meterologicos.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = dados_meterologicos.Interval()),
        inclusive = "left"
    )}
    
    i = 0
    for parametro in parametros_analisados:
        valores_parametro = dados_meterologicos.Variables(i).ValuesAsNumpy()[0]
        
        if type(valores_parametro) != np.float32:
            print(f"Tipo do parâmetro '{parametro}' antes da conversão: {type(valores_parametro)}")
            valores_parametro = pd.to_numeric(valores_parametro, errors='coerce').astype(np.float32)
                
            dados_meterologicos_format[parametro] = pd.to_numeric(dados_meterologicos.Variables(i), errors='coerce') 
        
        dados_meterologicos_format[parametro] = valores_parametro
        print(f"\nTipo do parâmetro '{parametro}': {type(dados_meterologicos_format[parametro])}")
            
        i = i + 1
    
    # Convertendo o dicionário para um DataFrame e definindo a coluna "date" como índice
    dados_meterologicos_format = pd.DataFrame(dados_meterologicos_format).set_index("date")
    dados_meterologicos_format.index = pd.to_datetime(dados_meterologicos_format.index)
    
    return dados_meterologicos_format
    
def construir_modelo():
    modelo = Sequential()
    modelo.add(SimpleRNN(32, input_shape=(10, 3), activation='relu'))
    modelo.add(Dense(1))
    
    modelo.compile(optimizer='adam', loss='mse')
    
    return modelo

if __name__ == "__main__":
    # Entrada de dados para treino
    localização = (-10.184, -48.333)
    data_inicio = "2023-01-01"
    data_fim = "2023-01-31"
    parametros_analisados =  [
        "temperature_2m_mean",
        "apparent_temperature_mean",
        "wind_speed_10m_max",
        "daylight_duration",
        "et0_fao_evapotranspiration"
    ]
    
    ativos = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
    
    dados = pegar_dados(localização, data_inicio, data_fim, parametros_analisados, ativos)
    
    dados["meterologia"] = preprocessar_dados(dados["meterologia"], parametros_analisados)
    
    print("\nDados das ações:")
    print(dados["ações"].columns)
    
    print("Tipos dos dados das ações:")
    print(dados["ações"].dtypes)
    
    print("\nDados metereológicos:")
    print(dados["meterologia"].columns)
    
    print("Tipos dos dados metereológicos:")
    print(dados["meterologia"].dtypes)