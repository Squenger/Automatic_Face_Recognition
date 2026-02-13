Reconnaissance Faciale Automatique
===================================

Bienvenue dans la documentation du projet de **Reconnaissance Faciale Automatique** développé par Aimine Meddeb et Leonard Beddouk.


Vue d'ensemble
--------------

Ce projet implémente un système complet de reconnaissance faciale utilisant les modèles ONNX de OpenCV (YuNet pour la détection, SFace pour la reconnaissance). Il permet de :

**Entraîner** le système sur des visages connus
**Identifier** automatiquement des personnes dans des photos
**Renommer** les fichiers avec les noms détectés
**Gérer** une base de données de visages
**Reconnaissance en temps réel** via webcam



Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Cloner le dépôt
   git clone <url-du-repo>
   cd Reconnaissance_Faciale_Automatique

   # Installer les dépendances
   uv sync

Lancement de l'Interface Graphique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv run main_qt

Ou double-cliquez sur ``Lancer_Interface.command`` (macOS).

Utilisation Basique
~~~~~~~~~~~~~~~~~~~

1. **Vérifier les modèles** : Cliquez sur "1. Vérifier Modèles"
2. **Entraîner** : Séléctionnez le dossier contenant les visages connus puis cliquez sur "2. Apprendre Visages"
3. **Traiter** : Séléctionnez le dossier contenant les visages inconnus puis cliquez sur "3. Lancer le Tri"
4. **Visualiser** : Cliquez sur "4. Voir les Résultats" pour observer les résultats

Documentation
----------------

.. toctree::
   :maxdepth: 2
   :caption: Guide Utilisateur

   guide_utilisateur
   structure_donnees

.. toctree::
   :maxdepth: 2
   :caption: Documentation Technique

   architecture
   api_reference
   tests



.. toctree::
   :maxdepth: 1
   :caption: Référence

   readme

Fonctionnalités Principales
----------------------------

Détection et Reconnaissance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **YuNet** : Détecteur de visages ultra-rapide et précis
- **SFace** : Reconnaissance faciale avec embeddings de 128 dimensions
- **Seuil ajustable** : Contrôle de la sensibilité de reconnaissance

Gestion de Base de Données
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Ajout de nouveaux visages par entraînement
- Suppression de personnes spécifiques
- Vidage complet de la base
- Sauvegarde automatique en format pickle

Interface Utilisateur
~~~~~~~~~~~~~~~~~~~~~

- **Interface PyQt6** moderne et intuitive
- **Progression en temps réel** avec callbacks
- **Visualisation des résultats** avec navigation
- **Ajustement du seuil** via slider

Mode Webcam
~~~~~~~~~~~

- Reconnaissance en temps réel
- Affichage des noms et scores de confiance
- Détection de multiples visages simultanément



Architecture
----------------

Le projet est structuré en 3 modules principaux :

``manager.py``
~~~~~~~~~~~~~~
Cœur du système gérant :

- Téléchargement et chargement des modèles ONNX
- Entraînement sur visages connus
- Traitement et reconnaissance d'images
- Gestion de la base de données

``main_qt.py``
~~~~~~~~~~~~~~
Interface graphique PyQt6 avec :

- Sélection de dossiers
- Contrôle du seuil de reconnaissance
- Affichage des logs en temps réel
- Visualisation des résultats

``webcam.py``
~~~~~~~~~~~~~
Mode temps réel pour :

- Capture vidéo depuis webcam
- Détection et reconnaissance continues
- Affichage des résultats sur la vidéo





Licence et Auteurs
---------------------

**Auteurs** : Aimine Meddeb, Leonard Beddouk

**Projet** : PAI 2025 - Institut d'Optique Graduate School

**Copyright** : © 2025
