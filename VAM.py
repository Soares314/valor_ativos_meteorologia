import openmeteo_requests
import yfinance as yf
from retry_requests import retry
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

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
    
    if isinstance(dados_acoes.columns, pd.MultiIndex): #Deixando apenas o valor de fechamento das ações
        dados_acoes = dados_acoes['Close']
    
    cols = dados_acoes.select_dtypes(include=["float64", "int64"]).columns
    dados_acoes[cols] = dados_acoes[cols].astype("float32")
    
    return {
            "meterologia": dados_meterologicos, 
            "ações": dados_acoes
           }    
    
def preprocessar_dados(dados_meterologicos, parametros_analisados):
    
    # Criando a estrutura base com o range completo de datas
    dados_meterologicos_format = { "date": pd.date_range(
        start = pd.to_datetime(dados_meterologicos.Time(), unit = "s"),
        end = pd.to_datetime(dados_meterologicos.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = dados_meterologicos.Interval()),
        inclusive = "left"
    )}
    
    i = 0
    for parametro in parametros_analisados:
        
        # Extraindo os valores de cada parâmetro
        valores_parametro = dados_meterologicos.Variables(i).ValuesAsNumpy()
        
        # Converte todo o array diretamente para float32 de forma segura
        dados_meterologicos_format[parametro] = valores_parametro.astype(np.float32)
        
        print(f"Parâmetro '{parametro}' carregado com sucesso. Dias: {len(valores_parametro)}")
            
        i = i + 1
    
    # Convertendo o dicionário com os vetores completos para DataFrame
    dados_meterologicos_format = pd.DataFrame(dados_meterologicos_format).set_index("date")
    dados_meterologicos_format.index = pd.to_datetime(dados_meterologicos_format.index)
    
    return dados_meterologicos_format
    
def criar_janela_tempo(dados_meterologicos, dados_acoes, janela_tempo=7):
    
    # Coletando os dados, baseado em uma janela de previsão
    X_seq, y_seq = [], []
    for i in range(len(dados_meterologicos) - janela_tempo):
        # Pega a janela de dias passados para as features
        X_seq.append(dados_meterologicos[i:(i + janela_tempo)])
        # Pega o preço do ativo no dia seguinte à janela
        y_seq.append(dados_acoes[i + janela_tempo])
        
        if i < 3:
            print("==== EXEMPLO ====")
            print("X (últimos dias):")
            print(dados_meterologicos[i:(i + janela_tempo)])
            print("y (dia seguinte):")
            print(dados_acoes[i + janela_tempo])
    
    print(f"Dados meterológicos: ")
    
    print(f"X_seq: {np.array(X_seq)}")
    print(f"y_seq: {np.array(y_seq)}")
    
    # sys.exit()
    return np.array(X_seq), np.array(y_seq)  

def construir_modelo(dados_meterologicos, dados_acoes, num_neuro1 = 64, num_neuro2 = 32, epochs = 50, batch_size = 32, porc_train = 0.8):
    
    #sys.exit()
    
    # Transformar preços brutos em Retornos Percentuais diários
    dados_acoes_retorno = dados_acoes.pct_change().dropna()
    print(f"Dados de ações de retorno: {dados_acoes_retorno.head()}")
    
    # Realinhar os dados após o dropna do retorno
    dados_comuns = dados_meterologicos.join(dados_acoes_retorno, how='inner').dropna()
    print(f"Colunas dos dados comuns: {dados_comuns.columns}")
    
    colunas_meterologicas = list(dados_meterologicos.columns)
    colunas_acoes = list(dados_acoes.columns)
    
    X_nao_norm = dados_comuns[colunas_meterologicas].values
    Y_nao_norm = dados_comuns[colunas_acoes].values

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler_X.fit_transform(X_nao_norm)
    y_scaled = scaler_Y.fit_transform(Y_nao_norm)
    
    X_seq, y_seq = criar_janela_tempo(X_scaled, y_scaled, janela_tempo=30)
    
    divisao = int(len(X_seq) * porc_train)

    X_train, X_test = X_seq[:divisao], X_seq[divisao:]
    y_train, y_test = y_seq[:divisao], y_seq[divisao:]
    
    modelo = Sequential([
        LSTM(units=num_neuro1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=num_neuro2, return_sequences=False),
        Dropout(0.2),
        Dense(units=y_train.shape[1])
    ])
    
    # Configura o otimizador com um valor inicial intermediário
    otimizador = Adam(learning_rate=0.001)
    modelo.compile(optimizer=otimizador, loss='mean_squared_error')

    # Configura o Callback de redução
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,         
        patience=5,         
        min_lr=0.0001,      
        verbose=1
    )

    modelo.compile(optimizer='adam', loss='mean_squared_error')
    modelo.summary()

    histórico_treinamento = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr],
        verbose=1
    )
    
    # Retornamos o scaler_Y para podermos desnormalizar os dados na avaliação
    return {
            "modelo": modelo,
            "histórico_treinamento": histórico_treinamento,
            "X_test": X_test,
            "y_test": y_test,
            "y_train": y_train,
            "scaler_Y": scaler_Y
           }
    
def salvar_modelo(modelo_treinado, nome_arquivo):
    
    # Salvando o modelo treinado com um timestamp para evitar sobrescrever modelos anteriores
    timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
    modelo_treinado.save(f"{nome_arquivo}_{timestamp}.keras")
    print(f"Modelo salvo em {nome_arquivo}")
    
def avaliar_modelo(modelo_treinado, scaler_Y, X_test, y_test, y_train):
    
    media = np.mean(y_train)

    y_baseline = np.full_like(y_test, media)
    
    y_pred_scaled = modelo_treinado.predict(X_test)
    
    # Desnormalizar os dados antes de calcular as métricas
    y_test_real = scaler_Y.inverse_transform(y_test)
    y_pred_real = scaler_Y.inverse_transform(y_pred_scaled)
    
    # MAE e MSE calculados nos valores reais de retorno
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)

    # Nota sobre o MAPE: Como retornos diários são frequentemente próximos de zero,
    # o MAPE pode explodir (divisão por quase zero). O R² e o MAE serão seus guias reais aqui.
    r2 = r2_score(y_test_real, y_pred_real)

    print("\n===  RESULTADOS ===")
    print("R² baseline:", r2_score(y_test, y_baseline))
    print(f"MAE : {mae:.6f} (Erro médio no retorno diário)")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²  : {r2:.6f}")
    
    # Plotar os resultados corrigidos
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_real, y_pred_real, alpha=0.5, color='blue', label='Predições')
    plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', label='Ideal')
    plt.xlabel("Retorno Real")
    plt.ylabel("Retorno Predito")
    plt.title("Predição vs Real (Retornos Diários - SLCE3)")
    plt.legend()

    out_path = 'resultados_corrigidos.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico salvo com sucesso em: {out_path}")
    

if __name__ == "__main__":
    # Entrada de dados para treino
    localização = ((-7.5325, -46.0356))
    data_inicio = "1995-01-01"
    data_fim = "2024-12-31"
    parametros_analisados =  [
        "temperature_2m_max",
        "temperature_2m_mean",
        "temperature_2m_min",

        "apparent_temperature_max",
        "apparent_temperature_mean",
        "apparent_temperature_min",

        "precipitation_sum",
        "rain_sum",
        "showers_sum",
        "snowfall_sum",

        "precipitation_hours",

        "weather_code",

        "sunshine_duration",
        "daylight_duration",

        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",

        "shortwave_radiation_sum",

        "et0_fao_evapotranspiration",
    ]
    
    ativos = ["AGRO3.SA"]
    
    # Pegando os dados metereológicos e das ações das suas APIS
    print("Pegando os dados metereológicos e das ações...")
    dados = pegar_dados(localização, data_inicio, data_fim, parametros_analisados, ativos)
    
    # Preprocessando os dados metereológicos
    print("Preprocessando os dados metereológicos...")
    dados["meterologia"] = preprocessar_dados(dados["meterologia"], parametros_analisados)
    
    print("\nShape das ações:")
    print(dados["ações"].shape)
    
    print("\nColunas das ações:")
    print(dados["ações"].columns)
    
    print("Tipos dos dados das ações:")
    print(dados["ações"].dtypes)
    
    print("\nValores das ações:")
    print(dados["ações"].head())
    
    
    print("\nDados metereológicos:")
    print(dados["meterologia"].columns)
    
    print("Tipos dos dados metereológicos:")
    print(dados["meterologia"].dtypes)
    
    print("\nShape dos dados metereológicos:")
    print(dados["meterologia"].shape)
    
    print("\nValores dos dados metereológicos:")
    print(dados["meterologia"].head())
    
    # Construindo o modelo LSTM
    print("Construindo o modelo LSTM...")
    modelo_resultado = construir_modelo(dados["meterologia"], dados["ações"])
    
    # Avaliando o modelo treinado e plotando os resultados
    print("Avaliando o modelo treinado e plotando os resultados...")
    avaliar_modelo(
        modelo_resultado["modelo"], 
        modelo_resultado["scaler_Y"], 
        modelo_resultado["X_test"], 
        modelo_resultado["y_test"],
        modelo_resultado["y_train"]
    )
    
    # Salvando o modelo treinado
    print("Salvando o modelo treinado...")
    salvar_modelo(modelo_resultado["modelo"], "modelo_VAM")
    
    