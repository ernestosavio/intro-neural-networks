import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # IIA - Trabajo Práctico 3 - Redes Neuronales
    """)
    return


@app.cell
def _():
    import os
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import joblib
    from joblib import Parallel, delayed


    import sklearn as skl
    from sklearn.neural_network import MLPRegressor
    from sklearn.neural_network import MLPClassifier
    from concurrent.futures import ProcessPoolExecutor
    from sklearn.metrics import mean_squared_error, zero_one_loss

    from copy import deepcopy

    return (
        MLPClassifier,
        MLPRegressor,
        Parallel,
        deepcopy,
        delayed,
        joblib,
        mean_squared_error,
        np,
        os,
        pd,
        plt,
        skl,
        zero_one_loss,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Explorando las Redes
    """)
    return


@app.cell
def _():
    # Parámetros
    """
    Comenzar y frenar el entrenamiento de una red es costoso, por lo que no
    mediremos las métricas una vez por época, sino que haremos varias épocas a la
    vez. Por ésto definimos sub-épocas, por ejemplo como 10, para realizar 10 épocas
    a la vez y luego medir las métricas de error.
    """
    sub_epocas = 25   # numero de epocas que entrena cada vez
    super_epocas = 100 # numero de veces que realizaremos sub-epocas
    # epocas ~= sub_epocas * super_epocas
    eta = 0.01        # learning rate
    alfa = 0.9        # momentum
    N2 = 60            # neuronas en la capa oculta
    return N2, alfa, eta, sub_epocas, super_epocas


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Definimos nuestras redes
    """)
    return


@app.cell
def _(MLPRegressor, N2, alfa, eta, sub_epocas):
    # Defino MLP para regresión
    regr = MLPRegressor(
        hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0,
        batch_size=1, learning_rate='constant', learning_rate_init=eta,
        momentum=alfa, nesterovs_momentum=False, tol=0.0, warm_start=True,
        max_iter=sub_epocas
    )

    print(regr)
    return


@app.cell
def _(MLPClassifier, N2, alfa, eta, sub_epocas):
    # Defino MLP para clasificación
    clasif = MLPClassifier(
        hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0,
        batch_size=1, learning_rate='constant', learning_rate_init=eta,
        momentum=alfa, nesterovs_momentum=False, tol=0.0, warm_start=True,
        max_iter=sub_epocas
    )

    print(clasif)
    return (clasif,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamos

    > **Warning**: el entrenamiento de las redes es computacionalmente pesado. Algunos de los entrenamientos que se plantean en el trabajo llevan varios minutos, dependiendo de la velocidad del procesador. El tiempo máximo en Google Colab fue de alrededor de 10 minutos. Tengan eso en cuenta para hacer el trabajo.
    """)
    return


@app.cell
def _(pd):
    def cargar_csv(path, xcols=2, separator=','):
        """
        Argumentos:
          path (str): ruta al archivo csv a cargar
          xcols (int): cantidad de columnas que representan las entradas,
            la columna restante representara la clase o dato de salida
        """
        df = pd.read_csv(path, header=None, sep=separator)
        X = df.loc[:, 0:(xcols-1)]
        y = df.loc[:, xcols]
        return X,y

    return (cargar_csv,)


@app.cell
def _(deepcopy, zero_one_loss):
    def entrenar_red(red, evaluaciones,
                     X_train, y_train,
                     X_val,   y_val,
                     X_test,  y_test):
        """
        Función que entrena una red ya definida previamente "evaluaciones" veces,
        cada vez entrenando un número de épocas elegido al crear la red y midiendo
        el error en train, validación y test al terminar ese paso de entrenamiento.
        Guarda y devuelve la red en el paso de evaluación que da el mínimo error de
        validación.

        Argumentos:
          red: red neuronal predefinida
          evaluaciones (int): las veces que evalua
          X_{}: los conjuntos de valores de entrada de train, validación y test
          y_{}: los conjuntos de valores de salida o clase

        Salidas:
          best_red: la red entrenada en el mínimo de validación
          error_{}: los errores de: train, validación y test medidos en cada
            evaluación
        """
        error_train = []
        error_val = []
        error_test = []
        best_val = 1.0
        best_red = red

        for epoch in range(evaluaciones):
          # red.partial_fit(X_train, y_train, classes=[0,1])
          ## Podríamos llamar partial_fit para realizar una sóla pasada a la vez,
          ## pero al ser muy costoso frenar y reanudar el entrenamiento realizamos
          ## varias épocas a la vez. Recordemos que la red fue definida con el
          ## parámetro 'sub-epocas', con lo cual cada llamado a 'fit' realiza esa
          ## cantidad de épocas
          red.fit(X_train, y_train)
          # error de training
          y_pred_train = red.predict(X_train)
          error_train.append(zero_one_loss(y_train, y_pred_train))
          # error de validacion
          y_pred_val = red.predict(X_val)
          cur_val = zero_one_loss(y_val, y_pred_val)
          error_val.append(cur_val)
          # error de test
          error_test.append(1 - red.score(X_test, y_test))
          if best_val > cur_val:
            best_val = cur_val
            best_red = deepcopy(red)
        return best_red, error_train, error_val, error_test

    return (entrenar_red,)


@app.cell
def _(cargar_csv, clasif, entrenar_red, skl, super_epocas):
    # Cargamos los datos...
    X, y = cargar_csv('./data/xor.csv')
    X2, X_test, y2, y_test = skl.model_selection.train_test_split(X, y, test_size=0.2)
    # Hacemos el split de training y testing
    X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X2, y2, test_size=0.2)
    # y spliteamos el training en training y validacion
    # Corremos el entrenamiento
    clasif_1, e_train, e_val, e_test = entrenar_red(clasif, super_epocas, X_train, y_train, X_val, y_val, X_test, y_test)  # para xor de 200 me quedan 40 de test  # para xor de 160 me quedan 32 de val
    return X_test, X_train, clasif_1, e_test, e_train, e_val, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ploteamos Algunas Métricas
    """)
    return


@app.cell
def _(e_test, e_train, e_val, np, plt, sub_epocas, super_epocas):
    rango = np.array(range(super_epocas)) * sub_epocas
    plt.plot(rango, e_train, label="train", linestyle=":")
    plt.plot(rango, e_val, label="validacion", linestyle="-.")
    plt.plot(rango, e_test, label="test", linestyle="-")
    plt.legend()
    plt.show()
    return


@app.cell
def _(X_test, clasif_1, plt):
    y_out = clasif_1.predict(X_test)
    X_out = X_test.copy()
    X_out[2] = y_out
    _c0 = X_out.loc[X_out[2] == 0]
    _c1 = X_out.loc[X_out[2] == 1]
    _xs0 = _c0[0].values.tolist()
    _ys0 = _c0[1].values.tolist()
    _xs1 = _c1[0].values.tolist()
    _ys1 = _c1[1].values.tolist()
    plt.scatter(_xs0, _ys0, color='red')
    plt.scatter(_xs1, _ys1, color='blue')
    plt.show()
    return


@app.cell
def _(X_train, plt, y_train):
    # diferenciar entre clase 0 y clase 1
    Xt = X_train.copy()
    Xt[2] = y_train
    _c0 = Xt.loc[Xt[2] == 0]
    _c1 = Xt.loc[Xt[2] == 1]
    _xs0 = _c0[0].values.tolist()
    _ys0 = _c0[1].values.tolist()
    _xs1 = _c1[0].values.tolist()
    _ys1 = _c1[1].values.tolist()
    plt.scatter(_xs0, _ys0, color='red')
    plt.scatter(_xs1, _ys1, color='blue')
    plt.show()
    return


@app.cell
def _(plt):
    def plot_classification(results, size_ds, feature_names, target_names):
        X_test, y_test = results["X_test"], results["y_test"]

        _, ax = plt.subplots(size_ds, 2, sharey=True, figsize=(20, 20), squeeze=False)

        for i in range(size_ds):
            # Ploteamos los datos reales a la izquierda
            scatter_true = ax[i, 0].scatter(X_test[:,0], X_test[:,1], c=y_test, s=10, cmap="bwr")
            ax[i, 0].set(xlabel=feature_names[0], ylabel=feature_names[1])
            _ = ax[i, 0].legend(
                scatter_true.legend_elements()[0], target_names, loc="lower right", title="Classes (True)"
            )

            # Ploteamos los datos predecidos a la derecha
            scatter_pred = ax[i, 1].scatter(X_test[:,0], X_test[:,1],
                                            c=results["nns"][i]["y_predict"], s=10, cmap="bwr")
            ax[i, 1].set(xlabel=feature_names[0], ylabel=feature_names[1])
            _ = ax[i, 1].legend(
                scatter_pred.legend_elements()[0], target_names, loc="lower right", title="Classes (Pred)"
            )


        plt.show()

    return (plot_classification,)


@app.cell
def _(np):
    def plot_errors(graph, training_error, testing_error, val_error, super_epocas, sub_epocas):
        rango = np.array(range(super_epocas)) * sub_epocas

        graph.plot(rango, training_error, label="train", linestyle=":")
        graph.plot(rango, testing_error, label="test", linestyle="-")
        graph.plot(rango, val_error, label="validation", linestyle="-.")

        #graph.xlabel('Epocas')
        #graph.ylabel('Error')

        graph.grid(True)
        graph.legend()
        # plt.figure(figsize=(8,5))
        # plt.ylim(0, 1)

        return graph

    return (plot_errors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Algunas funciones que fuimos usando
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funciones de entrenamiento
    """)
    return


@app.cell
def _(deepcopy, mean_squared_error, np):
    def entrenar_red_wd(
        red,
        evaluaciones,
        gamma,
        X_train,
        y_train,
        X_test,
        y_test,
        ord=1,
    ):
        """
        Función que entrena una red para regresión ya definida
        previamente "evaluaciones" veces,
        cada vez entrenando un número de épocas elegido al crear la red y midiendo
        el error en train, test y la suma de los pesos cada uno al cuadrado
        (o su valor absoluto, dependiendo del argumento ord) al terminar ese paso
        de entrenamiento.
        Guarda y devuelve la red en el paso de evaluación que da el mínimo error total
        (ErrorTrain + SumaCuadradaPesos).

        Argumentos:
          red: red neuronal predefinida
          evaluaciones (int): las veces que evalua
          X_{}: los conjuntos de valores de entrada de train, y test
          y_{}: los conjuntos de valores de salida o clase

        Salidas:
          best_red: la red entrenada en el mínimo error total
          error_{}: los errores de: train, test medidos en cada evaluación,
          la suma de los pesos cada uno al cuadrado (o su valor absoluto,
          dependiendo del argumento ord).
        """

        if (ord != 1) and (ord != 2):
            print("ord Incorrecto, utilice ord=1 o ord=2")
            return red, [], [], []

        error_train = []
        error_test = []
        norm_weights = []
        min_error = 1.0
        best_red = red

        for epoch in range(evaluaciones):
            red.fit(X_train, y_train)

            # Error de training
            y_pred_train = red.predict(X_train)
            e_train = mean_squared_error(y_train, y_pred_train)
            error_train.append(e_train)

            # Obtenemos los pesos de la red
            weights = red.coefs_

            # Obtenemos la norma 'ord' de los pesos en la evaluación
            # weights[0] -> Pesos que van desde la capa de entrada a la capa oculta
            # weights[1] -> Pesos que van desde la capa oculta a la capa de salida
            cur_norm_weights = np.linalg.norm(np.concatenate((weights[0].flatten(),
                                                              weights[1].flatten())
                                                            ), ord)
            if ord == 2:
                cur_norm_weights = cur_norm_weights * cur_norm_weights

            norm_weights.append(cur_norm_weights)

            # Error de test
            y_predict_test = red.predict(X_test)
            error_test.append(mean_squared_error(y_predict_test, y_test))

            # Error total
            total_error = e_train + gamma * cur_norm_weights

            if min_error > total_error:
                min_error = total_error
                best_red = deepcopy(red)
        return best_red, error_train, error_test, norm_weights

    return (entrenar_red_wd,)


@app.cell
def _(deepcopy, mean_squared_error):
    def entrenar_red_rgr(red, evaluaciones,
                         X_train, y_train,
                         X_val,   y_val,
                         X_test,  y_test):
        """
        Función que entrena una red ya definida previamente "evaluaciones" veces,
        cada vez entrenando un número de épocas elegido al crear la red y midiendo
        el error en train, validación y test al terminar ese paso de entrenamiento.
        Guarda y devuelve la red en el paso de evaluación que da el mínimo error de
        validación.

        Argumentos:
          red: red neuronal predefinida
          evaluaciones (int): las veces que evalua
          X_{}: los conjuntos de valores de entrada de train, validación y test
          y_{}: los conjuntos de valores de salida o clase

        Salidas:
          best_red: la red entrenada en el mínimo de validación
          error_{}: los errores de: train, validación y test medidos en cada
            evaluación
        """
        error_train = []
        error_val = []
        error_test = []
        best_val = 1.0
        best_red = red

        for epoch in range(evaluaciones):
          # Entrenamos la red
          red.fit(X_train, y_train)

          # Error de training
          y_pred_train = red.predict(X_train)
          error_train.append(mean_squared_error(y_train, y_pred_train))

          # Error de validacion
          y_pred_val = red.predict(X_val)
          cur_val = mean_squared_error(y_val, y_pred_val)
          error_val.append(cur_val)

          # Error de test
          y_pred_test = red.predict(X_test)
          error_test.append(mean_squared_error(y_pred_test, y_test))

          if best_val > cur_val:
            best_val = cur_val
            best_red = deepcopy(red)
        return best_red, error_train, error_val, error_test

    return (entrenar_red_rgr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ejercicio 1. Capacidad de Modelado.
    """)
    return


@app.cell
def _():
    # Incluimos la generación de espirales
    import spiral as spiral

    return (spiral,)


@app.cell
def _(MLPClassifier, entrenar_red, np, skl, spiral):
    def res_ej1():
        # Generamos los datos
        data = spiral.espirales(2600)
        X, y = np.vstack(data.input.values), data.output

        # Hacemos el split de training y testing (~600)
        X2, X_test, y2, y_test = skl.model_selection.train_test_split(X, y, test_size=0.23)

        # Spliteamos el training en training y validación
        X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X2, y2, test_size=0.20)

        # Parámetros 1
        sub_epocas = 20          # numero de epocas que entrena cada vez
        eval = 1000              # numero de veces que realizaremos sub-epocas
        # epocas ~= sub_epocas * super_epocas
        eta = 0.1                # learning rate
        alfa = 0.9               # momentum
        N2 = [2, 10, 20, 40]     # neuronas en la capa oculta

        res = []


        for i in range(len(N2)):

            clasif = MLPClassifier(
                hidden_layer_sizes=(N2[i],), activation='logistic', solver='sgd', alpha=0.0,
                batch_size=1, learning_rate='constant', learning_rate_init=eta,
                momentum=alfa, nesterovs_momentum=False, tol=0.0, warm_start=True,
                max_iter=sub_epocas,
            )

             # Corremos el entrenamiento
            clasif, e_train, e_val, e_test = entrenar_red(clasif, eval, X_train, y_train, X_val, y_val, X_test, y_test)  

            res.append( {"clasif"    : clasif,
                         "y_predict" : clasif.predict(X_test),
                         "e_train"   : e_train,
                         "e_val"     : e_val,
                         "e_test"    : e_test} )


        # Devolvemos un diccionario donde neurons y nns son una lista
        # de len(N2) dimensiones donde neurons[i] es la cantidad de
        # neuronas utilizadas en la capa escondida de la red neuronal nns[i]
        return { "X_test"  : X_test,
                 "y_test"  : y_test,
                 "nns"     : res 
               }

    return (res_ej1,)


@app.cell
def _(joblib, os, res_ej1):
    _archivo_cache = "resultados_ej1.pkl"

    if os.path.exists(_archivo_cache):
        ej1 = joblib.load(_archivo_cache)
    else:
        ej1 = res_ej1()
        joblib.dump(ej1, _archivo_cache)
    return (ej1,)


@app.cell
def _(ej1, plot_classification):
    # Ploteamos 
    size = len(ej1["nns"])
    plot_classification( ej1,
                         size,
                        ["rho", "theta"],
                        ["spiral 1", "spiral 2"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusión
    En las gráficas podemos ver que mientras incrementamos el numero de
    neuronas en la hidden layer obtenemos una mejor clasificación. Esto se
    debe, a que cuando combinamos sigmoids (las de la capa intermedia en el
    perceptron final) podemos aproximamos de mejor manera, es decir,
    obtenemos una mejor clasificación.

    Sin embargo, agregar neuronas en la capa intermedia puede provocar
    overfitting, ya que la red neuronal aprende ruido de los datos de
    entrenamiento. Pareciera no ser el caso de este ejercicio.

    Comparando con la clasificación hecha con arboles de decision vemos que,
    a pesar del tiempo de entrenamiento de las redes neuronales (que es
    mayor) se obtiene una peor clasificación. De esta manera, podemos decir
    que las redes neuronales son mas costosas que los arboles de decision.

    En nuestra opinion, la razón de la superioridad de arboles en la
    clasificación viene de que los arboles de decision son un método de
    aprendizaje para problemas discretos, en particular, clasificación.
    Por otro lado, la función MLPClassifier utiliza regresión y *softmax*
    para clasificar (herramientas para transformar un problema continuo
    en uno discreto), creemos que esta conversion lleva a una peor
    clasificación.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ejercicio 2. Mínimos Locales.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamos la red
    """)
    return


@app.cell
def _(cargar_csv, skl):
    def cargar_datos_ej2():
        # Cargamos los datos
        X, y = cargar_csv('./data/dos_elipses.data')
        X_test, y_test = cargar_csv('./data/dos_elipses.test')
        # Hacemos el split de training y testing
        X2, _, y2, _ = skl.model_selection.train_test_split(X, y, test_size=0.5)

        # spliteamos el training en training y validacion
        X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X2, y2, test_size=0.2)
        return X_test, X_train, X_val, y_test, y_train, y_val

    return (cargar_datos_ej2,)


@app.cell
def _(MLPClassifier, cargar_datos_ej2, entrenar_red):
    def res_ej2(eta, alfa):
        iter = 10

        # Parámetros 1
        # eta := learning rate | alfa := momentum
        sub_epocas = 50          # numero de epocas que entrena cada vez
        eval = 300              # numero de veces que realizaremos sub-epocas
        # epocas ~= sub_epocas * super_epocas
        N2 = 6     # neuronas en la capa oculta

        nns = []

        for i in range(iter):
            # Generamos los datos
            X_test, X_train, X_val, y_test, y_train, y_val = cargar_datos_ej2()

            clasif = MLPClassifier(
                hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0,
                batch_size=1, learning_rate='constant', learning_rate_init=eta,
                momentum=alfa, nesterovs_momentum=False, tol=0.0, warm_start=True,
                max_iter=sub_epocas
            )

             # Corremos el entrenamiento
            clasif, e_train, e_val, e_test = entrenar_red(clasif, eval, X_train, y_train, X_val, y_val, X_test, y_test)  

            nns.append( {"clasif"    : clasif,
                         "e_train"   : e_train,
                         "e_val"     : e_val,
                         "e_test"    : e_test})

        return { 
                 "nns" : nns,
                 "learning_rate" : eta,
                 "momentum" : alfa,
                 "super_epocas" : eval,
                 "sub_epocas" : sub_epocas
               }

    return (res_ej2,)


@app.cell
def _(Parallel, delayed, joblib, os, res_ej2):
    _archivo_cache = "resultados_ej2.pkl"

    if os.path.exists(_archivo_cache):
        ej2 = joblib.load(_archivo_cache)

    else:
        ej2 = {
                 "nns" : [],
                 "learning_rate": [],
                 "momentum": [],
                 "super_epocas" : [],
                 "sub_epocas" : []
              }

        ej2_cases = [
            (0.1, 0), (0.01, 0), (0.001, 0), (0.1, 0.5), (0.1, 0.9),
            (0.20, 0.9), (0.20, 0.5), (0.30, 0.9), (0.30, 0.5),
            (0.01, 0.5), (0.01, 0.9), (0.001, 0.5), (0.001, 0.9),
            (0.15, 0.5), (0.25, 0.5), (0.25, 0.75), (0.25, 0.25),
            (0.275, 0.25), (0.225, 0.25), (0.25, 0.20), (0.25, 0.30)
        ]

        # 1. Ejecutamos todos los casos en paralelo
        _res_paralelos = Parallel(n_jobs=-1)(
            delayed(res_ej2)(_eta, _alfa) for _eta, _alfa in ej2_cases
        )

        for _res in _res_paralelos:
            ej2["nns"].append(_res["nns"])
            ej2["learning_rate"].append(_res["learning_rate"])
            ej2["momentum"].append(_res["momentum"])
            ej2["super_epocas"].append(_res["super_epocas"])
            ej2["sub_epocas"].append(_res["sub_epocas"])

        joblib.dump(ej2, _archivo_cache)
    return (ej2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ploteamos los errores
    """)
    return


@app.cell
def _(ej2, np, pd):
    # Creamos y mostramos la tabla
    _aux_table_ej2 = {
                       "learning_rate": [],
                       "momentum": [],
                       "avg_e_test": [],
                       "avg_min_e_val": []
                     }



    for _i in range(len(ej2["nns"])):
        _min_errs_val = []
        _errs_test = []
        for _j in range(len(ej2["nns"][_i])):
            _min_e_val = min(ej2["nns"][_i][_j]["e_val"])
            _min_errs_val.append(_min_e_val)
            _k = ej2["nns"][_i][_j]["e_val"].index(_min_e_val)
            _errs_test.append(ej2["nns"][_i][_j]["e_test"][_k])

        _aux_table_ej2["avg_e_test"].append(np.mean(_errs_test))
        _aux_table_ej2["avg_min_e_val"].append(np.mean(_min_errs_val))

        _aux_table_ej2["momentum"].append(ej2["momentum"][_i])
        _aux_table_ej2["learning_rate"].append(ej2["learning_rate"][_i])

    table_ej2 = pd.DataFrame(_aux_table_ej2)
    table_ej2
    return (table_ej2,)


@app.cell
def _(ej2, np, plot_errors, plt, table_ej2):
    # Obtenemos el índice de la entrada del menor error promedio de validacion
    _k = table_ej2["avg_min_e_val"].idxmin()

    _e_train = []
    _e_val = []
    _e_test = []

    for _i in range(len(ej2["nns"][_k])):
        _e_train.append(ej2["nns"][_k][_i]["e_train"])
        _e_val.append(ej2["nns"][_k][_i]["e_val"])
        _e_test.append(ej2["nns"][_k][_i]["e_test"])

    _avg_e_train = np.mean(_e_train, axis=0)
    _avg_e_val = np.mean(_e_val, axis=0)
    _avg_e_test = np.mean(_e_test, axis=0)

    # Ploteamos los errores
    _, _ax = plt.subplots(1, 1, sharey=True, figsize=(15, 15), squeeze=False)

    plot_errors(_ax[0, 0], _avg_e_train,
                    _avg_e_test, _avg_e_val,
                    ej2["super_epocas"][_k], ej2["sub_epocas"][_k])

    _ax[0, 0].set_title("Errores")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusión
    Podemos concluir que los mejores valores de learning rate y momentum para
    esta red son los siguientes:
    Para obtener estos valores tuvimos en cuenta como afecta la modificacion
    de dichos parametros. Recordemos primero que implica cada parametro.

    **Learning rate**
    El learning rate es un valor que indica que tanta importancia le damos, al momento de ajustar
    los pesos, a los errores de aproximación (que tanto aprendemos de los errores).

    Un learning rate bajo hace que los pesos se ajusten de a poco. Esto puede provocar que el valor de los pesos se quede estancado en un mínimo local (ya que la derivada
    parcial es igual a 0). En contraposición, un learning rate alto ajusta los pesos de manera
    mas agresiva, esto puede provocar que los valores de los pesos se escapen o sobrepasen
    mínimos locales, pero puede que además sobrepasen al mínimo global.

    **Momentum**
    El momentum agrega un término en la corrección de los pesos de cada
    iteración del algoritmo (suma momentum * variacion_peso). Representa la inercia
    de la variación de los pesos. Esto logra que aún cayendo en un mínimo local (punto
    donde la la derivada parcial del error se hace 0) nos podamos salir del mismo
    (ya que este término no se anula por más que la derivada sea 0).

    Ahora bien, valores muy altos pueden generar que nos escapemos incluso del mínimo global.
    Por otro lado, un valor muy pequeño puede hacer que no sobrepasemos los minimos locales. Un
    valor adecuado nos ayudara a sobrepasar minimos locales y no escaparnos del minimo global.

    Volviendo a nuestro ejercicio, en los casos donde usamos momentum y
    learning rate muy alto tengan errores mucho mas altos que en otros casos.
    Lo mismo pasa en los casos con learning rate y momentum muy bajos.

    En los casos intermedios fuimos probando diferentes valores hasta que
    obtuvimos un error cercano al 10%. LR = 0.275, M = 0.25.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ejercicio 3. Regularización
    """)
    return


@app.cell
def _(cargar_csv, skl):
    def cargar_datos_ej3(val_per):
        # Cargamos los datos
        X, y = cargar_csv('./data/ikeda.data', 5, separator=r'\s+')

        # Spliteamos el training y validacion
        X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X, y, test_size=val_per)
        return X_train, X_val, y_train, y_val

    return (cargar_datos_ej3,)


@app.cell
def _(MLPRegressor, cargar_csv, cargar_datos_ej3, entrenar_red_rgr, skl):
    def res_ej3(val_perc):
        iter = 10

        # Parámetros
        eta = 0.01             # eta := learning rate
        alfa = 0.9             # alfa := momentum
        sub_epocas = 50        # numero de epocas que entrena cada vez
        eval = 400             # numero de veces que realizaremos sub-epocas
        # epocas ~= sub_epocas * super_epocas
        N2 = 30     # neuronas en la capa oculta

        # Cargamos los datos de test
        X, y = cargar_csv('./data/ikeda.test', xcols=5, separator=r'\s+')
        # Hacemos el split de testing (~ 2000 datos de train)
        _, X_test, _, y_test = skl.model_selection.train_test_split(X, y, test_size=0.416)

        train_perc = 1 - val_perc

        nns = []

        for i in range(iter):
            # Generamos los datos
            X_train, X_val, y_train, y_val = cargar_datos_ej3(val_perc)

            # Defino MLP para regresión
            regr = MLPRegressor(
                hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0,
                batch_size=1, learning_rate='constant', learning_rate_init=eta,
                momentum=alfa, nesterovs_momentum=False, tol=0.0, warm_start=True,
                max_iter=sub_epocas
            )

             # Corremos el entrenamiento
            regr, e_train, e_val, e_test = entrenar_red_rgr(regr, eval, X_train, y_train, X_val, y_val, X_test, y_test)

            nns.append( {"regr"    : regr,
                         "e_train"   : e_train,
                         "e_val"     : e_val,
                         "e_test"    : e_test})

        return { 
                 "nns" : nns,
                 "train_perc" : train_perc,
                 "val_perc" : val_perc,
                 "super_epocas" : eval,
                 "sub_epocas" : sub_epocas
               }

    return (res_ej3,)


@app.cell
def _(Parallel, delayed, joblib, os, res_ej3):
    _archivo_cache = "resultados_ej3.pkl"

    ej3_cases = [0.05, 0.25, 0.5]

    if os.path.exists(_archivo_cache):
        ej3 = joblib.load(_archivo_cache)
    else:

        ej3 = {
                "nnss" : [],
                "super_epocas" : [],
                "sub_epocas" : [],
                "train_perc" : [],
                "val_perc" : []
               }
        _res_paralelos = Parallel(n_jobs=-1)(
            delayed(res_ej3)(_c) for _c in ej3_cases
        )

        for _res in _res_paralelos:
            ej3["nnss"].append(_res["nns"])
            ej3["train_perc"].append(_res["train_perc"])
            ej3["val_perc"].append(_res["val_perc"])
            ej3["super_epocas"].append(_res["super_epocas"])
            ej3["sub_epocas"].append(_res["sub_epocas"])

        joblib.dump(ej3, _archivo_cache)
    return ej3, ej3_cases


@app.cell
def _(ej3, np, pd):
    # Creamos y mostramos la tabla para encontrar la mejor red
    _aux_table_ej3 = {
                       "train_perc" : [],
                       "val_perc" : [],
                       "avg_e_test": [],
                       "avg_min_e_val": []
                     }

    for _i in range(len(ej3["nnss"])):
        _min_errs_val = []
        _errs_test = []
        for _j in range(len(ej3["nnss"][_i])):
            _min_e_val = min(ej3["nnss"][_i][_j]["e_val"])
            _min_errs_val.append(_min_e_val)
            _k = ej3["nnss"][_i][_j]["e_val"].index(_min_e_val)
            _errs_test.append(ej3["nnss"][_i][_j]["e_test"][_k])

        _aux_table_ej3["avg_e_test"].append(np.mean(_errs_test))
        _aux_table_ej3["avg_min_e_val"].append(np.mean(_min_errs_val))

        _aux_table_ej3["val_perc"].append(ej3["val_perc"][_i])
        _aux_table_ej3["train_perc"].append(ej3["train_perc"][_i])

    table_ej3 = pd.DataFrame(_aux_table_ej3)
    table_ej3
    return


@app.cell
def _(ej3, ej3_cases, np, plot_errors, plt):
    _e_train = []
    _e_val = []
    _e_test = []

    _avg_e_train = []
    _avg_e_val = []
    _avg_e_test = []

    for _k in range(len(ej3_cases)):
        for _i in range(len(ej3["nnss"][_k])):
            _e_train.append(ej3["nnss"][_k][_i]["e_train"])
            _e_val.append(ej3["nnss"][_k][_i]["e_val"])
            _e_test.append(ej3["nnss"][_k][_i]["e_test"])

            _avg_e_train.append(np.mean(_e_train, axis=0))
            _avg_e_val.append(np.mean(_e_val, axis=0))
            _avg_e_test.append(np.mean(_e_test, axis=0))

    # Ploteamos los errores
    _, _ax = plt.subplots(len(ej3_cases), 1, sharey=True, figsize=(10, 10), squeeze=False)

    for _j in range(len(ej3_cases)):
        plot_errors(_ax[_j, 0], _avg_e_train[_j],
                        _avg_e_test[_j], _avg_e_val[_j],
                        ej3["super_epocas"][_j], ej3["sub_epocas"][_j])

        _ax[_j, 0].set_title(f"Errores - Validacion: {ej3["val_perc"][_j]} - Train: {ej3["train_perc"][_j]} ")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ejercicio 4. Regularización (2).
    """)
    return


@app.cell
def _(MLPRegressor, cargar_csv, entrenar_red_wd):
    def res_ej4(gamma, ord):
        # ord=1 : Suma de los valores absolutos de los pesos
        # ord=2 : Suma de los cuadrados de los pesos

        iter = 10

        eta = 0.05
        alfa = 0.3
        sub_epocas = 20
        eval = 4000 # "super epocas"
        N2=6

        # Cargamos los datos de test
        X_test, y_test = cargar_csv('./data/ssp.test', 12)

        nns = []

        for i in range(iter):
            # Defino MLP para regresión
            regr = MLPRegressor(hidden_layer_sizes=(N2,), activation='logistic',
                                solver='sgd', alpha=gamma,
                                batch_size=1, learning_rate='constant',
                                learning_rate_init=eta, momentum=alfa,
                                nesterovs_momentum=False, tol=0.0,
                                warm_start=True, max_iter=sub_epocas)

            # Cargamos los datos de entrenamiento
            X_train, y_train = cargar_csv('./data/ssp.data', 12)

            # Corremos el entrenamiento
            regr, e_train, e_test, norm_weight = entrenar_red_wd(regr, eval, gamma, 
                                                                  X_train, y_train,
                                                                  X_test, y_test,
                                                                  ord)
            nns.append( {"regr"    : regr,
                         "e_train"   : e_train,
                         "norm_weight"     : norm_weight,
                         "e_test"    : e_test})

        return { 
                 "nns" : nns,
                 "ord"  : ord,
                 "gamma" : gamma,
                 "super_epocas" : eval,
                 "sub_epocas" : sub_epocas
               }

    return (res_ej4,)


@app.cell
def _(Parallel, delayed, joblib, os, res_ej4):
    _archivo_cache = "resultados_ej4.pkl"
    ej4_cases = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    if os.path.exists(_archivo_cache):
        ej4 = joblib.load(_archivo_cache)
    else:
        ej4 = { 
                "nnss": [],
                "ord": [],
                "gamma": [],
                "super_epocas": [],
                "sub_epocas": []
              }

        # 1. Ejecutamos todos los casos en paralelo
        _resultados_paralelos = Parallel(n_jobs=-1)(
            delayed(res_ej4)(_c, 2) for _c in ej4_cases
        )

        for _res in _resultados_paralelos:
            ej4["nnss"].append(_res["nns"])
            ej4["gamma"].append(_res["gamma"])
            ej4["ord"].append(_res["ord"])
            ej4["super_epocas"].append(_res["super_epocas"])
            ej4["sub_epocas"].append(_res["sub_epocas"])

        joblib.dump(ej4, _archivo_cache)
    return ej4, ej4_cases


@app.cell
def _(np):
    def calc_total_errors(e_trains, norm_weights, gamma, ord):
        e_trains = np.array(e_trains)
        norm_weights = np.array(norm_weights)

        if (ord == 1):
            total_errs = e_trains + gamma * norm_weights
        else: 
            total_errs = e_trains + gamma * (norm_weights * norm_weights)

        return total_errs.tolist()

    return (calc_total_errors,)


@app.cell
def _(calc_total_errors, ej4, np, pd):
    # Creamos y mostramos la tabla para encontrar la mejor red
    _aux_table_ej4 = {
                       "ord" : [],
                       "gamma" : [],
                       "avg_e_test": [],
                       "avg_e_train": [],
                       "avg_min_total_err": []
                     }

    _total_errors = []

    for _i in range(len(ej4["nnss"])):
        _min_total_errs = []
        _errs_test = []
        _errs_train = []
    
        for _j in range(len(ej4["nnss"][_i])):
            _total_errors = calc_total_errors(ej4["nnss"][_i][_j]["e_train"], 
                                             ej4["nnss"][_i][_j]["norm_weight"],
                                             ej4["gamma"][_i],
                                             ej4["ord"][_i])
            _min_total_err = min(_total_errors)
            _min_total_errs.append(_min_total_err)

            _k = _total_errors.index(_min_total_err)
            _errs_train.append(ej4["nnss"][_i][_j]["e_train"][_k])
            _errs_test.append(ej4["nnss"][_i][_j]["e_test"][_k])

        _aux_table_ej4["avg_e_test"].append(np.mean(_errs_test))
        _aux_table_ej4["avg_min_total_err"].append(np.mean(_min_total_errs))
        _aux_table_ej4["avg_e_train"].append(np.mean(_errs_train))
    
        _aux_table_ej4["gamma"].append(ej4["gamma"][_i])
        _aux_table_ej4["ord"].append(ej4["ord"][_i])

    table_ej4 = pd.DataFrame(_aux_table_ej4)
    table_ej4
    return


@app.cell
def _(ej4, ej4_cases, np, plot_errors, plt):
    # Graficamos los errores de la mejor red
    _e_train = []
    _norm_weight = []
    _e_test = []

    _avg_e_train = []
    _avg_norm_weight = []
    _avg_e_test = []

    for _k in range(len(ej4_cases)):
        for _i in range(len(ej4["nnss"][_k])):
            _e_train.append(ej4["nnss"][_k][_i]["e_train"])
            _norm_weight.append(ej4["nnss"][_k][_i]["e_val"])
            _e_test.append(ej4["nnss"][_k][_i]["e_test"])

            _avg_e_train.append(np.mean(_e_train, axis=0))
            _avg_norm_weight.append(np.mean(_norm_weight, axis=0))
            _avg_e_test.append(np.mean(_e_test, axis=0))

    # Ploteamos los errores
    _, err_graph_ej4 = plt.subplots(len(ej4_cases), 1, sharey=True, figsize=(10, 10), squeeze=False)

    # Graficamos la penalizacion de la mejor red
    _, pen_graph_ej4 = plt.subplots(len(ej4_cases), 1, sharey=True, figsize=(10, 10), squeeze=False)

    for _j in range(len(ej4_cases)):
        plot_errors(_ax[_j, 0], _avg_e_train[_j],
                        _avg_e_test[_j], _avg_e_val[_j],
                        ej4["super_epocas"][_j], ej4["sub_epocas"][_j])

        _ax[_j, 0].set_title(f"Errores - Validacion: {ej4["val_perc"][_j]} - Train: {ej4["train_perc"][_j]} ")
    plt.show()
    return


@app.cell
def _():
    # Graficamos la penalizacion de la mejor red
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ejercicio 5. Dimensionalidad.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # (OPCIONAL) Ejercicio 6. Multiclase.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # (OPCIONAL) Ejercicio 7. Minibatch.
    """)
    return


if __name__ == "__main__":
    app.run()
