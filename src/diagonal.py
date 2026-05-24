import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Generic, TypeVar
from typing import Any, Optional
from typing import List, Tuple

def diagonales(d: int, C: float, n: int):
    """
    Genera sampleos de dos gausianas d-dimensionales en posición diagonal
    (una respecto a la otra), con dispersión C * sqrt(d).
    Cada sampleo posee n/2 puntos.

    Argumentos:
        d: número de dimensiones
        C: constante de ajuste de dispersión
        n: número de ejemplos a generar

    Retorna:
        X: matriz con d columnas y n filas de datos
        y: arreglo de n elementos con las clases correspondientes
            a cada fila de datos
    """
    
    X = np.array([])  

    center0 = [-1 for i in range(d)]
    center1 = [ 1 for i in range(d)]
 
    datos0 = np.random.normal(center0, C * np.sqrt(d), (n//2, d)) # Quiero generar n//2 puntos randoms de d dimensiones, siguiendo una distribución normal de media centro0, y desviación estándar C * Sqrt(d)
    datos1 = np.random.normal(center1, C * np.sqrt(d), (n//2, d)) # Devuelve arrays

    
    y0 = [0 for i in range(n//2)]
    y1 = [1 for i in range(n//2)]

    X = np.concatenate (([datos0, datos1])) # Para poder concatenar bien los datos
    y = np.concatenate([y0,  y1])


    return pd.DataFrame({'input': X.tolist(), 'output': y})

def test_diagonales(d: int, C: float, n: int, m: int):
  test = []

  for i in range(m):
    # Generamos los datos de test:
    test.append(diagonales(d, C, n))
    
  return test