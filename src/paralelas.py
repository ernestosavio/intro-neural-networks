import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Generic, TypeVar
from typing import Any, Optional
from typing import List, Tuple

def paralelas(d: int, C: float, n: int):

    X, y = np.array([]), np.array([])

    center0 = [0 for i in range(d)]
    center1 = [0 for i in range(d)]

    center0[0] = 1
    center1[0] = -1

    datos0 = np.random.normal(center0, C, (n//2, d)) 
    datos1 = np.random.normal(center1, C, (n//2, d)) 


    y0 = [0 for i in range(n//2)]
    y1 = [1 for i in range(n//2)]

    X = np.concatenate (([datos0, datos1])) # Para poder concatenar bien los datos
    y = np.concatenate([y0,  y1])


    return pd.DataFrame({'input': X.tolist(), 'output': y})

def test_paralelas(d: int, C: float, n: int, m: int):
  test = []

  for i in range(m):
    # Generamos los datos de test:
    test.append(paralelas(d, C, n))
    
  return test
