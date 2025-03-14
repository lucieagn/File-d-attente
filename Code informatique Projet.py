# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:04:50 2023

@author: Guillaume, Lucie, Yannick
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as sp

#%% Simulation d'une file d'attente M/M/1 à nombre de clients fixe

def queueMM1_clients_fixes(lbd = 1, mu = 1, nbc = 100, ct0 = 0) :
    #Simulation de la loi exponentielle pour les durées d'arrivée avec cumul pour les temps d'arrivée
    explbdraw=-np.log(rand(nbc))/lbd # durée séparant les arrivées de chaque client
    explbd = np.append(np.zeros(ct0),explbdraw) # les clients initiaux arrivent au temps initial
    cumlbd = np.cumsum(explbd) # cumul pour obtenir les temps d'arrivée
    
    #Simulation de la loi exponentielle pour les durées de service pour chaque client
    expmu = -np.log(rand(nbc+ct0))/mu #durée de traitement de chaque client
    
    #Temps auquel chaque client est traité (temps de sortie de la file d'attente)
    traite=np.array([i for i in expmu]) # Le temps de sortie est au moins égal à la durée du service de chaque client
    traite[0]+=cumlbd[0] #Le client initial n'attend pas
    for i in range(1,nbc+ct0) :
        if cumlbd[i] < traite[i-1] : #Si un client est en cours de traitement,
        #le nouveau client sort de la file quand le client devant lui est traité + sa durée de service
            traite[i]+=traite[i-1]
        else : #Si aucun client n'est en cours de traitement, celui qui arrive sort de la file quand il arrive + service
            traite[i]+=cumlbd[i]
    
    prec=round(traite[-1]*200) # ajuste la précision en fonction du temps de traitement du dernier client
    temps = np.linspace(0,traite[-1]*prec/(prec-1), prec+1)
    #En faisant ça, on a le temps de traitement du dernier client en avant-dernier terme et la file est vidée à la fin
    
    clients = np.zeros(len(temps))
    clients[0]=ct0 #Clients de départ
    for t in range(1,len(temps)) :
        clients[t]=sum((cumlbd<temps[t]).astype(int))-sum((traite<temps[t]).astype(int))
        #Le nombre de clients présents dans la file est la différence des clients arrivés et de ceux sortis de celle-ci.
    return temps, clients

#Affichage du graphe pour des paramètres donnés ci-dessous
lbd = .5 ; mu = 1 ; nb_clients = 100 ; clients_init = 50

t1,c1 = queueMM1_clients_fixes(lbd, mu, nb_clients, clients_init)
plt.figure(1)
plt.plot(t1,c1)
plt.plot(np.array([t1[0],t1[-1]]),np.array([0,0]),'r')
plt.xlabel("Temps")
plt.ylabel("Nombre de clients")
plt.suptitle("MM1 : Clients base / rajoutés : "+str(clients_init)+" / "+str(nb_clients)+
             ", Arrivée ~ℰ("+str(lbd)+"), Gestion ~ℰ("+str(mu)+")")
plt.show()

#%% Simulation d'une file d'attente M/M/1 à temps fixe

def queueMM1_temps_fixe(lbd = 1, mu = 1, tps = 100, ct0 = 0) :
    #Simulation de la loi exponentielle pour les durées d'arrivée tant que le temps max n'est pas atteint
    if ct0>0 :
        cumlbd = np.zeros(ct0)
        #Les clients présents au départ arrivent en temps nul
    else :
        cumlbd = np.array([-np.log(rand())/lbd])
        #S'il n'y a pas de client, on laisse le premier client arriver
    while cumlbd[-1]<tps :
        cumlbd = np.append(cumlbd,cumlbd[-1]-np.log(rand())/lbd)
        #Les clients affluent tant que le temps max n'est pas atteint
    
    prec=round(tps*200) # ajuste la précision en fonction du temps défini
    temps = np.linspace(0, tps, prec)
    
    #Temps de traitement du premier client : il n'attend pas
    traite=np.array([cumlbd[0]-np.log(rand())/mu])
    
    #De la même façon, tant que le temps max n'est pas atteint, on traite des clients
    i=1
    while traite[-1] < tps :
        if cumlbd[i] < traite[i-1] :
            traite = np.append(traite, traite[i-1]-np.log(rand())/mu)
        else :
            traite = np.append(traite, cumlbd[i]-np.log(rand())/mu)
        i=i+1
    
    clients = np.zeros(prec)
    clients[0] = ct0 #Clients de départ
    for t in range(1,prec) :
        clients[t]=sum((cumlbd<temps[t]).astype(int))-sum((traite<temps[t]).astype(int))
        #Le nombre de clients présents dans la file est la différence des clients arrivés et de ceux sortis.
    
    return temps, clients

#Affichage du graphe pour des paramètres donnés ci-dessous
lbd = 1 ; mu = .8 ; temps = 250 ; clients_init = 60

t2,c2 = queueMM1_temps_fixe(lbd, mu, temps, clients_init)
plt.figure(2)
plt.plot(t2,c2)
plt.xlabel("Temps")
plt.ylabel("Nombre de clients")
plt.suptitle("MM1 : Clients t0 : "+str(clients_init)+", temps = "+str(temps)+
             ", Arrivée ~ℰ("+str(lbd)+"), Gestion ~ℰ("+str(mu)+")")
plt.show()

#%% Simulation d'une file d'attente M/M/∞ à temps fixe

def queueMMinf_temps_fixe(lbd=1,mu=1,tps=100,ct0=0) :
    #De la même façon : simulation de la loi exponentielle pour les durées d'arrivée avec cumul pour les temps d'arrivée
    if ct0>0 :
        cumlbd = np.zeros(ct0)
    else :
        cumlbd = np.array([-np.log(rand())/lbd])
    while cumlbd[-1]<tps :
        cumlbd = np.append(cumlbd,cumlbd[-1]-np.log(rand())/lbd)
    
    prec=round(tps*200) # ajuste la précision en fonction du temps défini
    temps = np.linspace(0, tps, prec)
    
    #Temps de traitement pour chaque client
    traite = cumlbd - np.log(rand(len(cumlbd)))/mu
    
    clients = np.zeros(prec)
    clients[0]=ct0 #Clients de départ
    for t in range(1,prec) :
        clients[t]=sum((cumlbd<temps[t]).astype(int))-sum((traite<temps[t]).astype(int))
        #Le nombre de clients présents dans la file est la différence des clients arrivés et de ceux traités.
    
    return temps, clients

#Affichage du graphe pour des paramètres donnés ci-dessous
lbd = 5 ; mu = .05 ; temps = 250 ; clients_init = 200

t3,c3 = queueMMinf_temps_fixe(lbd, mu, temps, clients_init)
plt.figure(3)
plt.plot(t3,c3)
plt.xlabel("Temps")
plt.ylabel("Nombre de clients")
plt.suptitle("Clients t0 : "+str(clients_init)+", temps = "+str(temps)+
             ", Arrivée ~ℰ("+str(lbd)+"), Gestion ~ℰ("+str(mu)+")")
plt.show()

#%% Simulation de la figure 11.1 en bas du PDF

temps = 60 ; clients_init = 100 ; lbd = 1

plt.figure(4)

mu1 = 3
t4,c4 = queueMM1_temps_fixe(lbd, mu1, temps, clients_init)
plt.plot(t4, c4, label = "M/M/1")

mu2 = .5
t5,c5 = queueMMinf_temps_fixe(lbd, mu2, temps, clients_init)
plt.plot(t5, c5, label = "M/M/∞")

plt.xlabel("Temps")
plt.ylabel("Nombre de clients")
plt.suptitle("Clients t0 : "+str(clients_init)+", temps = "+str(temps)+
             ", Arrivée ~ℰ("+str(lbd)+"), Gestion ~ℰ("+str(mu1)+"(MM1) / "+str(mu2)+"(MM∞))")
plt.legend()
plt.show()

#%% Quantité de clients arrivés à un temps T

def clients(tps = 100, lbd = 1) :
    c = 0
    s = -np.log(rand())/lbd
    while s < tps :
        #Incrémente jusqu'à atteindre le temps demandé
        s-=np.log(rand())/lbd
        c+=1
    return c

print(clients(tps = 100, lbd = .5))
print(clients(tps = 100, lbd = 1))
print(clients(tps = 100, lbd = 2))
print(clients(tps = 50, lbd = 1))

#%% Quantité de clients dans la file d'attente à un temps T

def clinqueue_MMinf(tps = 100, lbd = 1, mu = 1, ct0 = 0) :
    if tps < 0 :
        #Les clients arrivent en temps positif
        return 0
    clitps = 0
    for i in range(ct0) :
        if -np.log(rand())/mu > tps :
            clitps+=1
        #Si les clients présents au départ ne sont pas encore traités, ils sont comptés
    
    #Temps d'arrivée du premier client dans la file d'attente et temps de départ de celui-ci
    cliarr = -np.log(rand())/lbd
    cliserv = cliarr - np.log(rand())/mu
    if cliarr <= tps and tps < cliserv :
        clitps+=1
        #Si au temps donné, le premier client est arrivé dans la file d'attente mais n'en est pas sorti, il est compté
    
    while cliarr < tps or cliserv < tps :
        #Si les clients continuent d'affluer ou d'être traités jusqu'au temps donné, on les prend en compte
        cliarr = cliarr - np.log(rand())/lbd
        cliserv = cliarr - np.log(rand())/mu
        #Temps d'arrivée et de sortie du nouveau client
        if cliarr <= tps and tps < cliserv :
            clitps+=1
            #Si le temps donné est entre le temps d'arrivée du nouveau client et son temps de sortie, il est compté
    return clitps

print(clinqueue_MMinf(tps = 100, lbd = 1, mu = .02, ct0 = 0))
print(clinqueue_MMinf(tps = 100, lbd = 1, mu = .02, ct0 = 100))
print(clinqueue_MMinf(tps = 20, lbd = 1, mu = .02, ct0 = 100))
print(clinqueue_MMinf(tps = 2, lbd = 1, mu = .5, ct0 = 100))

#%% Simulation de X_t

def binom(n,p) : #Simulation d'une loi binomiale
    return sum((rand(n)<p).astype(int))

def pois(lbd) : #Simulation d'une loi de Poisson
    s = 0
    n = -1
    while s<=1 :
        n+=1
        s-=np.log(rand())/lbd
    return n

def X_t(lbd, mu, ct0, t) :
    #X_t est défini comme un produit de convolution entre les deux
    #Donc il est simulé comme la somme des deux lois
    return binom(ct0, np.exp(-mu * t)) + pois((lbd/mu)*(1-np.exp(-mu * t)))

print(X_t(lbd = 1, mu = .02, ct0 = 0, t = 100))
print(X_t(lbd = 1, mu = .02, ct0 = 100, t = 100))
print(X_t(lbd = 1, mu = .02, ct0 = 100, t = 20))
print(X_t(lbd = 1, mu = .5, ct0 = 100, t = 2))


#%% Probabilités exactes associées (en utilisant scipy.stats et pas de simulation)

def pbin(k, n, p) : 
    return sp.binom.pmf(k, n, p)

def ppoi(k, lbd) :
    return sp.poisson.pmf(k,lbd)

def proba_X_t(lbd, mu, ct0, t, n) :
    som = 0
    if n < 0 or t < 0 :
        return 0
        #Les clients ne sont ni en nombre négatif ni n'arrivent en temps négatif
    else :
        for m in range(min(ct0,n)+1) :
            # P(X=n) = somme(m=-inf -> +inf) P(B = m) * P(P=n-m)
            # avec B la loi binomiale et P la loi de Poisson
            # m est dans Z tout entier mais la loi binomiale est définie sur {0, ... , k}
            # et la loi de Poisson est définie sur N donc n-m≥0 donc m≤n
            # donc m≥0 et m≤n et m≤k donc m est compris entre 0 et min(n,k) (+1 car Python)
            som += pbin(m, ct0, np.exp(-mu * t)) * ppoi(n - m, lbd/mu * (1 - np.exp(-mu * t) ) )
        return som

#%% Probas pour lbd = 1 ; mu = .02
# Plus le nombre de clients au départ est faible, plus le temps de calcul est rapide
# puisque le min entre ct0 et n est faible dans le range de m juste au-dessus.

lbd = 1 ; mu = .02 ; clients_init = 0 ; tmax = 251

T = np.array([i for i in range(tmax)])
N = np.array([i for i in range(30,105)])
P = np.array([[proba_X_t(lbd, mu, clients_init, t, n) for t in T] for n in N])
#On affiche les probas pour tous les temps et tous les nombres de clients tant qu'elle est pas quasi nulle
#(ici, la proba avec ces paramètres d'avoir moins de 30 clients est quasi nulle)

t0 = 5
#On part de t0=5 parce qu'on est p.s. au nombre de clients initiaux en temps 0
#Et les probas diminuent drastiquement après peu de temps

Tp = np.array([i for i in range(t0, tmax)])
PP = np.array([P[i,t0:] for i in range(len(P))])

fig = plt.figure(6)
ax = plt.axes(projection = '3d')

TT,NN=np.meshgrid(Tp,N)
surf = ax.plot_surface(TT, NN, PP, cmap = cm.hsv, linewidth = 0, antialiased = False)
ax.set_xlabel('Temps', fontsize = 20)
ax.set_ylabel('Nb de clients', fontsize = 20)

cbar = fig.colorbar(surf, ax = ax)
cbar.set_label(label = 'Probabilités', size = 20)
cbar.ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
plt.legend()
plt.show()


#%% Probas pour lbd = 5 ; mu=.05 (mêmes commentaires)

lbd = 5 ; mu = .05 ; clients_init = 200 ; tmax = 151 #(~ 5 min de traitement pour clients_init = 200)

T = np.array([i for i in range(tmax)])
N = np.array([i for i in range(60,210)])
P = np.array([[proba_X_t(lbd, mu, clients_init, t, n) for t in T] for n in N])

t0 = 5

Tp = np.array([i for i in range(t0, tmax)])
PP = np.array([P[i,t0:] for i in range(len(P))])

fig = plt.figure(7)
ax = plt.axes(projection ='3d')

fs = 17 ; fst = 14
TT,NN=np.meshgrid(Tp,N)
surf = ax.plot_surface(TT, NN, PP, cmap = cm.hsv, linewidth = 0, antialiased = False)
ax.set_xlabel('Temps', fontsize = fs)
ax.set_ylabel('Nb de clients', fontsize = fs)

cbar = fig.colorbar(surf, ax = ax)
cbar.set_label(label = 'Probabilités', size = fs)
cbar.ax.tick_params(axis = 'both', which = 'major', labelsize = fst)

ax.tick_params(axis = 'both', which = 'major', labelsize = fst)
plt.legend()
plt.show()

