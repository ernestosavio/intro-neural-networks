import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn import tree as skltree
from sklearn import datasets as skldatasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

import diagonal as diagonal
import paralelas as paralelas

# Definimos funciones comunes para los ejercicios:
def predict_clf(clf, ds):
  # Obtenemos el dataset
  X = np.vstack(ds.input.values)
  y = ds.output

  # Predecimos los resultados
  predict_result = clf.predict(X)

  # Calculamos la pérdida
  loss = round(zero_one_loss(y, predict_result), 2)
  acc = round(1-loss, 2)

  # Devolvemos el resultado
  return {
        "dataset": ds,
        "prediction": predict_result,
        "loss": loss,
        "accuracy": acc
    }


def generate_trained_clfs(train_dss):
  clfs = []

  for i in range(len(train_dss)):
    # Creamos un árbol de decisiones
    clf = skltree.DecisionTreeClassifier(criterion="entropy",
                                       min_impurity_decrease=0.002,
                                       random_state=0,
                                       min_samples_leaf=5)

    # Obtenemos los conjuntos de datos del dataset
    X_train, y_train = np.vstack(train_dss[i].input.values), train_dss[i].output

    # Entrenamos el árbol
    clf.fit(X_train, y_train)

    # Agregamos el árbol entrenado a la lista
    clfs.append(clf)

  return clfs

def trained_clfs_results(clfs, train_dss, test_ds):
  clf_results = []

  for i in range(len(clfs)):
    # Predecimos los resultados para el train_dataset i
    train_result = predict_clf(clfs[i], train_dss[i])

    # Predecimos los resultados para el test_dataset i
    test_result = predict_clf(clfs[i], test_ds)

    # Agregamos los resultados a la lista
    clf_results.append({
            "clf": clfs[i],
            "test_result": test_result,
            "train_result": train_result,
            "nodes": clfs[i].tree_.node_count
        })

  return clf_results

def average_results(results):
    # Calculamos los valores promedios
    avg_train_loss = np.mean([r["train_result"]["loss"] for r in results])
    avg_test_loss = np.mean([r["test_result"]["loss"] for r in results])
    avg_nodes = np.mean([r["nodes"] for r in results])

    # Devolvemos los valores
    return {
        "train_loss": avg_train_loss,
        "test_loss": avg_test_loss,
        "nodes": avg_nodes
    }

def print_predict_result(train_err, train_acc, test_err, test_acc, t):
  print(f'Tamaño del árbol: {t}')
  print(f'Train loss: {train_err}')
  print(f'Train accuracy: {train_acc}\n')
  print(f'Test loss: {test_err}')
  print(f'Test accuracy: {test_acc}\n')


def run_experiment(type_generator, size_ds, d, C, m, N):
  # Generamos el caso de test
  if type_generator == "diagonal":
    test_dataset = diagonal.diagonales(d, C, N)
  elif type_generator == "paralelas":
    test_dataset = paralelas.paralelas(d, C, N)

  training_avg_error = []
  testing_avg_error = []
  tree_avg_size = []
  clf_resultss = []

  # Generamos los casos de prueba para cada n
  for i in range(len(size_ds)):
    # Generamos los m casos de entrenamiento
    if type_generator == "diagonal":
      training_datasets = diagonal.test_diagonales(d, C, size_ds[i], m)
    elif type_generator == "paralelas":
      training_datasets = paralelas.test_paralelas(d, C, size_ds[i], m)

    # Generamos los m árboles ya entrenados
    clfs = generate_trained_clfs(training_datasets)

    # Calculamos los errores en cada árbol con respecto a los datasets de
    # entrenamiento y el dataset de test
    clf_results = trained_clfs_results(clfs, training_datasets, test_dataset)

    # Calculamos los valos promedios del error de test, del error de
    # entrenamiento y del tamaño del árbol
    avg_results = average_results(clf_results)
    training_avg_error.append(avg_results["train_loss"])
    testing_avg_error.append(avg_results["test_loss"])
    tree_avg_size.append(avg_results["nodes"])

    # Imprimimos los resultados de cada entrenamiento
    # print(f'Largo conjunto ENTRENAMIENTO: {size_ds[i]}')
    # for j in range(len(clf_results)):
    #   print_predict_result(clf_results[j]["train_result"]["loss"],
    #                        clf_results[j]["train_result"]["accuracy"],
    #                        clf_results[j]["test_result"]["loss"],
    #                        clf_results[j]["test_result"]["accuracy"],
    #                        clf_results[j]["nodes"])

    clf_resultss.append(clf_results)

  # Devolvemos los resultados promedios
  return {
      "test_dataset": test_dataset,
      "clf_resultss": clf_resultss,
      "avg_train_loss": training_avg_error,
      "avg_test_loss": testing_avg_error,
      "avg_nodes": tree_avg_size
  }


def plot_classification(results, size_ds, feature_names, target_names):
  test_dataset = results["test_dataset"]
  clf_results = results["clf_resultss"]

  X_test, y_test = np.vstack(test_dataset.input.values), test_dataset.output


  _, ax = plt.subplots(len(size_ds), 2, sharey=True, figsize=(20, 20))

  for i in range(len(size_ds)):
    # Ploteamos los datos reales a la izquierda
    scatter_true = ax[i, 0].scatter(X_test[:,0], X_test[:,1], c=y_test, s=10)
    ax[i, 0].set(xlabel=feature_names[0], ylabel=feature_names[1])
    _ = ax[i, 0].legend(
        scatter_true.legend_elements()[0], target_names, loc="lower right", title="Classes (True)"
    )

    # Ploteamos los datos predecidos a la derecha
    scatter_pred = ax[i, 1].scatter(X_test[:,0], X_test[:,1],
                                    c=clf_results[i][0]["test_result"]["prediction"], s=10)
    ax[i, 1].set(xlabel=feature_names[0], ylabel=feature_names[1])
    _ = ax[i, 1].legend(
        scatter_pred.legend_elements()[0], target_names, loc="lower right", title="Classes (Pred)"
    )



def plot_errors(results, size_ds):
    training_avg_error = results["avg_train_loss"]
    testing_avg_error = results["avg_test_loss"]

    xG = np.concatenate([size_ds, size_ds])
    yG = np.concatenate([training_avg_error, testing_avg_error])
    cG = np.concatenate([np.zeros(len(size_ds), dtype=int),
                         np.ones(len(size_ds), dtype=int)])

    plt.figure(figsize=(8,5))
    plt.plot(size_ds, training_avg_error, marker = 'o', label = 'Train')
    plt.plot(size_ds, testing_avg_error, marker = 'o', label = 'Test')
    plt.xlabel('tamaño dataset')
    plt.ylabel('avg error')
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()

def plot_nodes(results, size_ds):
    plt.figure(figsize=(10, 5))
    plt.plot(size_ds, results["avg_nodes"], marker = 'o')
    plt.xlabel('tamaño dataset')
    plt.ylabel('tree avg size')
    plt.xscale('log')
    plt.grid()
    plt.show()
