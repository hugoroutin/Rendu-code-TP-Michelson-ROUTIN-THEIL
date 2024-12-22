# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:05:08 2024

@author: routin 

To run a cell, press shift+enter while in focus of a cell.
"""
#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def afine(x,a,b):
    return a*x+b

def modele_contraint(x, a):
    return a * x + (y0 - a * x0)

#%%
x=np.linspace(1,8,8)

m2= np.array([30.74,31.78,32.61,33.56,34.95,35.65,36.65,37.70])
m2=m2-30.31 #passage de la position absolue à la distance e

xErrorValues = np.ones([1,8])*0.01
yErrorValues = np.ones([8,])*0.5 #erreur de \pm 0.5 anneaux

print(np.shape(yErrorValues))
print(np.shape(m2))

params_x, cov =curve_fit(afine, x, m2,[1,0],sigma=list(yErrorValues) )


plt.scatter( m2,x, color='black',label="données mesurées" )
plt.plot( afine(x, *params_x),x, label="Fit afine ", color='blue')

plt.errorbar(m2, x, xerr = xErrorValues, yerr = yErrorValues, capsize = 1,
  ecolor = 'red', marker = 'o', markersize = 1, markerfacecolor = 'blue', linestyle = 'none')
plt.title("Profil et Fit du nb d'anneaux en fonction de la distance ")
plt.xlabel("e (Lecture du vernier) (mm)")
plt.ylabel("Nombres d'anneaux")
plt.legend()
plt.xlim(0,9)
plt.show()


print(np.sqrt(cov[0, 0]))
print(params_x)

#%%


p=np.array([-2,-1,0,1,2])
dixm=np.array([26.85,28.60,30.31,32.15,34.75])
quinzem=np.array([28.6,29.26,30.31,31.5,32.05])
dixm=dixm-30.31
quinzem=quinzem-30.31
pErrorValues = np.array([0.4,0.25,0.1,0.25,0.4])
yErrorValues = np.ones([5,])*0.04

# print(np.shape(p))
# print(np.shape(yErrorValues))

params_10m, cov =curve_fit(afine, p,dixm,[1,0],sigma=list(yErrorValues) )
params_15m, cov =curve_fit(afine, quinzem,p,[1,0],sigma=list(yErrorValues) )


plt.scatter( p,dixm, color='black',label="données mesurées pour diam=10mm" )
plt.plot(p, afine(p, *params_10m), label="Fit afine 10mm", color='blue')

plt.errorbar(p, dixm, xerr = pErrorValues, yerr = yErrorValues, capsize = 1,ecolor = 'red', marker = 'o', markersize = 1, markerfacecolor = 'blue', linestyle = 'none')
plt.scatter( p,quinzem, color='green',label="données mesurées pour diam=15mm" )
plt.plot(p, afine(p, *params_15m), label="Fit afine 15mm", color='blue')

plt.errorbar(p, quinzem, xerr = pErrorValues, yerr = yErrorValues, capsize = 1,ecolor = 'red', marker = 'o', markersize = 1, markerfacecolor = 'blue', linestyle = 'none')
plt.title("Position des annulations de contraste")
plt.xlabel("Ordre d'interférence p")
plt.ylabel("e")
plt.legend()
plt.xlim(-2.5,2.5)
plt.grid()
plt.show()


#%%
p=np.array([-2,-1,0,1,2])

dixm=np.array([26.85,28.60,30.31,32.15,34.75])
quinzem=np.array([28.6,29.26,30.31,31.5,32.05])
dixm=dixm-30.31
quinzem=quinzem-30.31
pErrorValues = np.array([0.4,0.25,0.1,0.25,0.4])
yErrorValues = np.ones([5,])*0.04

y0,x0=0,0 #définition du point de passage obligatoire des droites

def modele_contraint(x, a):
    return a * x + (y0 - a * x0)

params_10m, cov =curve_fit(modele_contraint, p,dixm,[10],sigma=list(pErrorValues) )
a_opti=params_10m
b_opti=y0-a_opti*x0
params_10m=np.append(params_10m,b_opti)
print('Pente pour 10mm',a_opti)

plt.figure(figsize=(10, 10))  # Largeur = 10, Hauteur = 5
#plt.subplot(1, 2, 1)
plt.scatter( p,dixm, color='gold',label="Pour diam=10mm",s=100 ,edgecolors='black')
plt.plot(p, afine(p, *params_10m), label="Fit afine 10mm", color='gold',linewidth=2.5)
plt.errorbar(p, dixm, xerr = pErrorValues, yerr = yErrorValues, capsize = 1,ecolor = 'red', marker = 'o', markersize = 1, markerfacecolor = 'blue', linestyle = 'none')
erreur_pente=np.sqrt(cov[0, 0])
print('erreur pente pour 10mm',erreur_pente)

plt.plot(p, afine(p,a_opti+erreur_pente,b_opti),linestyle='dotted',  color='forestgreen',linewidth=2)
plt.plot(p, afine(p,a_opti-erreur_pente,b_opti),linestyle='dotted',  color='darkblue',linewidth=2)


plt.title("Position des annulations de contraste pour diam=10mm")
plt.xlabel("Ordre d'interférence p")
plt.ylabel("e")
plt.legend()
plt.xlim(-2.5,2.5)
plt.ylim(-4.5,4.7)
plt.grid()


params_15m, cov =curve_fit(modele_contraint, p,quinzem,[10],sigma=list(pErrorValues) )
a_opti=params_15m
b_opti=y0-a_opti*x0
params_15m=np.append(params_15m,b_opti)
print('Pente pour 15mm',a_opti)

plt.grid()

plt.scatter( p,quinzem, color='deeppink',label="Pour diam=15mm", s=100 ,edgecolors='black')
plt.plot(p, afine(p, *params_15m), label="Fit afine 15mm", color='deeppink',linewidth=2.5)
plt.errorbar(p, quinzem, xerr = pErrorValues, yerr = yErrorValues, capsize = 1,ecolor = 'red', marker = 'o', markersize = 1, markerfacecolor = 'blue', linestyle = 'none')
erreur_pente=np.sqrt(cov[0, 0])
print('erreur pente pour 15mm',erreur_pente)

plt.plot(p, afine(p,a_opti+erreur_pente,b_opti),linestyle='dotted',  color='forestgreen',linewidth=2,label="Pente + incertitude sur la pente")
plt.plot(p, afine(p,a_opti-erreur_pente,b_opti),linestyle='dotted',  color='darkblue',linewidth=2,label="Pente - incertitude sur la pente")


plt.title("Position des annulations de contraste pour différents diamètres",fontsize=18)
plt.xlabel("Ordre d'interférence p",fontsize=15)
plt.ylabel("e (mm)",fontsize=15)
plt.legend(fontsize=14)
plt.xlim(-2.5,2.5)
plt.ylim(-4.5,4.7)
plt.grid()
plt.show()

#%%


import numpy as np
import matplotlib.pyplot as plt

# Constantes physiques
h = 6.626e-34  # Constante de Planck (J.s)
c = 3e8         # Vitesse de la lumière (m/s)
k = 1.38e-23    # Constante de Boltzmann (J/K)

# Fonction de Planck pour la densité spectrale d'émission d'un corps noir
def loi_de_planck(longueur_d_onde, T):
    """
    Calcule la densité spectrale d'émission d'un corps noir.

    Paramètres :
        longueur_d_onde : ndarray
            Longueur d'onde en mètres.
        T : float
            Température en kelvins.

    Retourne :
        ndarray : Densité spectrale d'émission (W/m^3).
    """
    return (2 * h * c**2 / longueur_d_onde**5) / (np.exp(h * c / (longueur_d_onde * k * T)) - 1)

# Sensibilité de l'oeil humain (courbe de luminosité photopique)
def sensibilite_oeil(longueur_d_onde_nm):
    """
    Approximation de la sensibilité de l'oeil humain en fonction de la longueur d'onde.

    Paramètres :
        longueur_d_onde_nm : ndarray
            Longueur d'onde en nanomètres.

    Retourne :
        ndarray : Sensibilité relative (de 0 à 1).
    """
    # Données approchées pour la courbe de luminosité
    # Source : CIE photopic luminosity function (approximée)
    return 1.019 * np.exp(-0.5 * ((longueur_d_onde_nm - 555) / 55)**2)

# Plage des longueurs d'onde
longueur_d_onde_nm = np.linspace(280, 1400, 800)  # Longueurs d'onde en nanomètres
longueur_d_onde_m = longueur_d_onde_nm * 1e-9    # Conversion en mètres

# Température du corps noir
T = 3000  # Température en kelvins

# Calcul des densités spectrales d'émission et de sensibilité
densite_corps_noir = loi_de_planck(longueur_d_onde_m, T)
sensibilite = sensibilite_oeil(longueur_d_onde_nm)

# Normalisation des données pour le tracé
densite_corps_noir_normalisee = densite_corps_noir / np.max(densite_corps_noir)
sensibilite_normalisee = sensibilite / np.max(sensibilite)

# Tracé des courbes
plt.figure(figsize=(10, 6))

# Courbe du corps noir
plt.plot(longueur_d_onde_nm, densite_corps_noir_normalisee, label="Densité spectrale d'émission (3000 K)", color="red",linewidth=3.5)

# Courbe de sensibilité de l'œil
plt.plot(longueur_d_onde_nm, sensibilite_normalisee, label="Sensibilité de l'œil humain", color="green",linewidth=3.5)

# Configuration du graphique
plt.title("Sensibilité de l'œil humain et densité spectrale d'un corps noir (3000 K)",fontsize=15)
plt.xlabel("Longueur d'onde (nm)",fontsize=12)
plt.ylabel("Valeur normalisée")
plt.legend(fontsize=15)
plt.grid(True)
plt.xlim(280, 1400)

# Affichage
plt.show()
