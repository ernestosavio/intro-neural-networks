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
    import joblib
    import os
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import sklearn as skl
    from sklearn.neural_network import MLPRegressor
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import mean_squared_error, zero_one_loss

    from copy import deepcopy

    return (
        MLPClassifier,
        MLPRegressor,
        deepcopy,
        joblib,
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
    def cargar_csv(path, xcols=2):
        """
        Argumentos:
          path (str): ruta al archivo csv a cargar
          xcols (int): cantidad de columnas que representan las entradas,
            la columna restante representara la clase o dato de salida
        """
        df = pd.read_csv(path, header=None)
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
def _(np, plt):
    def plot_errors(training_error, testing_error, val_error, epocas):
        rango = np.array(range(epocas))

        plt.plot(rango, training_error, label="train", linestyle=":")
        plt.plot(rango, testing_error, label="test", linestyle="-")
        plt.plot(rango, val_error, label="validation", linestyle="-.")

        plt.xlabel('Epocas')
        plt.ylabel('Error')
        plt.grid()
        plt.legend()
        plt.show()
        # plt.figure(figsize=(8,5))
        # plt.ylim(0, 1)
    return


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
                random_state = 0
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


@app.cell
def _(cargar_csv, skl):
    def cargar_datos_ej2(seed):
        # Cargamos los datos
        X, y = cargar_csv('./data/dos_elipses.data')
        X_test, y_test = cargar_csv('./data/dos_elipses.test')
        # Hacemos el split de training y testing
        X2, _, y2, _ = skl.model_selection.train_test_split(X, y, test_size=0.5, random_state = seed)

        # spliteamos el training en training y validacion
        X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X2, y2, test_size=0.2, random_state = seed)
        return X_test, X_train, X_val, y_test, y_train, y_val

    return (cargar_datos_ej2,)


@app.cell
def _(MLPClassifier, cargar_datos_ej2, entrenar_red, np):
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
            X_test, X_train, X_val, y_test, y_train, y_val = cargar_datos_ej2(i)

            clasif = MLPClassifier(
                hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0,
                batch_size=1, learning_rate='constant', learning_rate_init=eta,
                momentum=alfa, nesterovs_momentum=False, tol=0.0, warm_start=True,
                max_iter=sub_epocas, random_state=i
            )

             # Corremos el entrenamiento
            clasif, e_train, e_val, e_test = entrenar_red(clasif, eval, X_train, y_train, X_val, y_val, X_test, y_test)  

            nns.append( {"clasif"    : clasif,
                         "y_predict" : clasif.predict(X_test),
                         "e_train"   : e_train,
                         "e_val"     : e_val,
                         "e_test"    : e_test} )

        avg_e_train = np.mean([nn["e_train"] for nn in nns])
        avg_e_val = np.mean([nn["e_val"] for nn in nns])
        avg_e_test = np.mean([nn["e_test"] for nn in nns])

        return { "nns" : nns,
                 "avg_e_train"  : avg_e_train,
                 "avg_e_val"  : avg_e_val,
                 "avg_e_test"     : avg_e_test,
                 "learning_rate" : eta,
                 "momentum" : alfa
               }

    return (res_ej2,)


@app.cell
def _(joblib, os, res_ej2):
    _archivo_cache = "resultados_ej2.pkl"

    if os.path.exists(_archivo_cache):
        table_ej2 = joblib.load(_archivo_cache)

    else:
        table_ej2 = {
             "avg_e_train": [],
             "avg_e_val": [],
             "avg_e_test": [],
             "learning_rate": [],
             "momentum": []
        }

        _casos = [
            (0.1, 0), (0.01, 0), (0.001, 0), (0.1, 0.5), (0.1, 0.9),
            (0.20, 0.9), (0.20, 0.5), (0.30, 0.9), (0.30, 0.5),
            (0.01, 0.5), (0.01, 0.9), (0.001, 0.5), (0.001, 0.9),
            (0.15, 0.5), (0.25, 0.5), (0.25, 0.75), (0.25, 0.25),
            (0.275, 0.25), (0.225, 0.25), (0.25, 0.20), (0.25, 0.30)
        ]

        for _eta, _alfa in _casos:
            _res = res_ej2(_eta, _alfa)
            table_ej2["nns"].append(_res["nns"])
            table_ej2["avg_e_train"].append(_res["avg_e_train"])
            table_ej2["avg_e_val"].append(_res["avg_e_val"])
            table_ej2["avg_e_test"].append(_res["avg_e_test"])
            table_ej2["learning_rate"].append(_res["learning_rate"])
            table_ej2["momentum"].append(_res["momentum"])

        joblib.dump(table_ej2, _archivo_cache)
    return (table_ej2,)


@app.cell
def _(pd, table_ej2):
    # Creamos y mostramos la tabla
    show_table_ej2 = pd.DataFrame(table_ej2)
    show_table_ej2 = show_table_ej2.drop("nns", axis=1)
    
    show_table_ej2
    return


@app.cell
def _():
    # Ploteamos los errores


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
    def cargar_datos_ej3(val_per, seed):
        # Cargamos los datos
        X, y = cargar_csv('./data/ikeda.data')

        # Spliteamos el training y validacion
        X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X, y, test_size=val_per, random_state = seed)
        return X_train, X_val, y_train, y_val

    return (cargar_datos_ej3,)


@app.cell
def _(MLPRegressor, cargar_csv, cargar_datos_ej3, entrenar_red, skl):
    def res_ej3():
        # Parámetros
        eta = 0.01             # eta := learning rate
        alfa = 0.9             # alfa := momentum
        sub_epocas = 50          # numero de epocas que entrena cada vez
        eval = 400              # numero de veces que realizaremos sub-epocas
        # epocas ~= sub_epocas * super_epocas
        N2 = 30     # neuronas en la capa oculta

        val_percs = [0.05, 0.25, 0.5]
        nns = []

        seed = 0

        # Cargamos los datos de test
        X, y = cargar_csv('./data/ikeda.test')
        # Hacemos el split de testing (~ 2000 datos de train)
        _, X_test, _, y_test = skl.model_selection.train_test_split(X, y, test_size=0.416, seed = seed)

        for i in range(len(val_percs)):
            val_perc = val_percs[i]
            train_perc = 1 - val_perc

            # Generamos los datos
            X_train, X_val, y_train, y_val = cargar_datos_ej3(val_perc, seed)

            # Defino MLP para regresión
            regr = MLPRegressor(
                hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0,
                batch_size=1, learning_rate='constant', learning_rate_init=eta,
                momentum=alfa, nesterovs_momentum=False, tol=0.0, warm_start=True,
                max_iter=sub_epocas
            )

             # Corremos el entrenamiento
            regr, e_train, e_val, e_test = entrenar_red(regr, eval, X_train, y_train, X_val, y_val, X_test, y_test)  

            nns.append( {"nn"        : regr,
                         "y_predict" : regr.predict(X_test),
                         "train_perc" : train_perc,
                         "val_perc" : val_perc,
                         "e_train"   : e_train,
                         "e_val"     : e_val,
                         "e_test"    : e_test} )


        return {
                "X_test" : X_test,
                 "y_test" : y_test,
                 "nns"         : nns,
                 "train_perc" : eta,
                 "val_perc"   : alfa
               }

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ejercicio 4. Regularización (2).
    """)
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
