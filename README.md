# README - Simulation de Files d'Attente M/M/1 et M/M/∞

## 1. Introduction
Ce projet s'intéresse à la modélisation mathématique des files d'attente à l'aide d'outils probabilistes. Les files d’attente sont omniprésentes dans les systèmes modernes et permettent d’analyser l'efficacité des services tels que les caisses en supermarché, les serveurs informatiques ou encore les réseaux de télécommunication.

Nous nous focalisons ici sur deux types de files d'attente :
- **M/M/1** : Une seule file avec un unique serveur, où les temps entre arrivées et les durées de service suivent des lois exponentielles.
- **M/M/∞** : Chaque client dispose d’un serveur immédiatement, et donc aucun temps d’attente n’existe.

## 2. Prérequis
Avant d'exécuter ce projet, assurez-vous d'avoir installé les bibliothèques suivantes :

```bash
pip install numpy matplotlib scipy
```

## 3. Structure du projet
Le projet contient plusieurs parties, incluant la simulation des files d'attente M/M/1 et M/M/∞, ainsi que la visualisation des résultats sous forme de graphes.

Les fichiers principaux sont :
- `queueMM1_clients_fixes(lbd, mu, nbc, ct0)`: Simulation d’une file M/M/1 avec un nombre de clients fixes.
- `queueMM1_temps_fixe(lbd, mu, tps, ct0)`: Simulation d’une file M/M/1 pour une durée fixe.
- `queueMMinf_temps_fixe(lbd, mu, tps, ct0)`: Simulation d’une file M/M/∞ pour une durée fixe.

Chaque fonction modélise l'arrivée et le traitement des clients en utilisant des lois exponentielles.

## 4. Utilisation

### Exécution des simulations

Dans un script Python, vous pouvez exécuter une simulation comme suit :

```python
import matplotlib.pyplot as plt
from queue_simulation import queueMM1_temps_fixe

# Paramètres
lbd = 1  # Taux d'arrivée
mu = 0.8  # Taux de service
temps = 250  # Durée de la simulation
clients_initiaux = 60

# Exécution
t, c = queueMM1_temps_fixe(lbd, mu, temps, clients_initiaux)

# Affichage du résultat
plt.figure()
plt.plot(t, c)
plt.xlabel("Temps")
plt.ylabel("Nombre de clients")
plt.title(f"File M/M/1 avec lambda={lbd}, mu={mu}")
plt.show()
```

### Comparaison entre M/M/1 et M/M/∞

Il est possible de comparer les deux modèles en les superposant :

```python
import matplotlib.pyplot as plt
from queue_simulation import queueMM1_temps_fixe, queueMMinf_temps_fixe

lbd = 1  # Taux d'arrivée
temps = 60  # Durée
t0 = 100  # Nombre initial de clients

# Simulation
t1, c1 = queueMM1_temps_fixe(lbd, 3, temps, t0)
t2, c2 = queueMMinf_temps_fixe(lbd, 0.5, temps, t0)

# Affichage
plt.figure()
plt.plot(t1, c1, label="M/M/1")
plt.plot(t2, c2, label="M/M/∞")
plt.xlabel("Temps")
plt.ylabel("Nombre de clients")
plt.legend()
plt.title("Comparaison des files d'attente")
plt.show()
```

## 5. Interprétation des Résultats
- **M/M/1** : La file fluctue en fonction de l'arrivée et du traitement des clients. Un taux d'arrivée plus grand que le taux de service provoque un engorgement.
- **M/M/∞** : Chaque client est pris en charge immédiatement, donc la file représente directement le nombre de clients en service à chaque instant.

## 6. Conclusion
Ce projet permet de mieux comprendre le comportement des systèmes de files d’attente et d'illustrer les concepts abordés dans les processus stochastiques. Il peut être étendu à d'autres modèles plus complexes (M/M/c, M/G/1, etc.) pour analyser différents scénarios de gestion des flux de clients.


Ce projet a été réalisé dans le cadre du cours de processus stochastiques M1 IM, Pr. Roland Diel.
Licence : MIT.

