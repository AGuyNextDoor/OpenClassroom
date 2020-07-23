import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def multi_plotbox(data, classes_size = 500.):

  taille_classe = float(classes_size) # taille des classes pour la discrétisation
  groupes = data # va recevoir les données agrégées à afficher

  # on calcule des tranches allant de 0 au solde maximum par paliers de taille taille_classe
  # tranches = np.arange(0, 100, taille_classe)
  # tranches += taille_classe/2 # on décale les tranches d'une demi taille de classe
  # indices = np.digitize(100, tranches) # associe chaque solde à son numéro de classe

  for key, value in groupes: # pour chaque tranche, ind reçoit le numéro de tranche et tr la tranche en question
      montants = groupes[key] # sélection des individus de la tranche ind
      if len(montants) > 0:
          g = {
              'valeurs': montants,
              'centre_classe': value-(taille_classe/2),
              'taille': len(montants),
              'quartiles': [np.percentile(montants,p) for p in [25,50,75]]
          }
          groupes.append(g)

  # affichage des boxplots
  plt.boxplot([g["valeurs"] for g in groupes],
              positions= [g["centre_classe"] for g in groupes], # abscisses des boxplots
              showfliers= False, # on ne prend pas en compte les outliers
              widths= taille_classe*0.7, # largeur graphique des boxplots
              manage_xticks= False)

  # affichage des effectifs de chaque classe
  for g in groupes:
      plt.text(g["centre_classe"],0,"(n={})".format(g["taille"]),horizontalalignment='center',verticalalignment='top')
  plt.show()

  # affichage des quartiles
  for n_quartile in range(3):
      plt.plot([g["centre_classe"] for g in groupes],
               [g["quartiles"][n_quartile] for g in groupes])
  plt.show()
