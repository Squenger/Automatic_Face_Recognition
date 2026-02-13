Structure des Données
======================

Ce document explique comment le système organise et stocke les données.

Vue d'Ensemble
--------------

Le système utilise trois types de données :

1. **Images d'entraînement** : Visages connus pour l'apprentissage
2. **Images à traiter** : Photos à identifier
3. **Base de données** : Encodages des visages connus

Dossier ``known_faces/``
-------------------------

Organisation
~~~~~~~~~~~~

.. code-block:: text

   known_faces/
   ├── Personne_1/
   │   ├── photo1.jpg
   │   ├── photo2.png
   │   └── photo3.jpeg
   ├── Personne_2/
   │   └── image.jpg
   └── Personne_3/
       ├── selfie1.jpg
       └── selfie2.jpg

Règles
~~~~~~

- **Un dossier = Une personne**
- Le nom du dossier devient l'identifiant
- Plusieurs photos par personne (recommandé : 1-5)
- Formats acceptés : ``.jpg``, ``.jpeg``, ``.png``

Exemples de Noms
~~~~~~~~~~~~~~~~

**Bons exemples** :

- ``Aimine_Meddeb``
- ``Leonard_Beddouk``


**À éviter** :

- ``Photo 1`` (trop générique)
- ``Aimine Meddeb`` (espace, préférer underscore)
- ``Jean`` (il peut y avoir plusieurs personnes avec le même nom)

Dossier ``unknown_faces/``
---------------------------

Organisation
~~~~~~~~~~~~

.. code-block:: text

   unknown_faces/
   ├── vacances_2023_001.jpg
   ├── DSC_0982.png
   ├── random_img.jpeg
   └── photo_groupe.jpg

Caractéristiques
~~~~~~~~~~~~~~~~

- Noms de fichiers quelconques
- Seront renommés automatiquement après traitement
- Peuvent contenir un ou plusieurs visages

Résultat Après Traitement
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   unknown_faces/
   ├── Aimine_Meddeb.jpg          # 1 personne reconnue
   ├── Leonard_Beddouk.png        # 1 personne reconnue
   ├── Aimine_Meddeb_Leonard_Beddouk.jpeg  # 2 personnes


Gestion des Collisions
^^^^^^^^^^^^^^^^^^^^^^^

Si un fichier avec le même nom existe déjà :

.. code-block:: text

   Aimine_Meddeb.jpg      # Original
   Aimine_Meddeb_2.jpg    # Première collision
   Aimine_Meddeb_3.jpg    # Deuxième collision

Dossier ``encodings_data/``
----------------------------

Généré Automatiquement
~~~~~~~~~~~~~~~~~~~~~~

Ce dossier est créé automatiquement lors du premier entraînement.

.. warning::
   Ne modifiez PAS manuellement les fichiers de ce dossier ! Vous pouvez le vider avec le bouton "Vider la base de données" dans l'interface utilisateur.

Contenu
~~~~~~~

``visages_connus.pkl``
^^^^^^^^^^^^^^^^^^^^^^

**Format** : Pickle Python (binaire)

**Contenu** : Tuple de deux listes

.. code-block:: python

   (
       [feature_vector_1, feature_vector_2, ...],  # Vecteurs de 128 dimensions
       ["Aimine_Meddeb", "Leonard_Beddouk", ...]   # Noms correspondants
   )

**Utilisation** : Chargé en mémoire pour la reconnaissance (fais partie du processus de reconnaissance)

``visages_connus_noms.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Format** : Texte brut

**Contenu** : Liste des noms (un par ligne)

.. code-block:: text

   Aimine_Meddeb
   Aimine_Meddeb
   Leonard_Beddouk
   Marie_Curie

.. note::
   Les noms peuvent apparaître plusieurs fois (une fois par photo d'entraînement).

**Utilisation** : Référence rapide pour vérifier les personnes enregistrées

Dossier ``models_onnx/``
-------------------------

Modèles de Deep Learning
~~~~~~~~~~~~~~~~~~~~~~~~~

``face_detection_yunet_2023mar.onnx``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Taille** : ~232 KB
- **Fonction** : Détection de visages
- **Sortie** : Coordonnées des boîtes englobantes + points de repère pour faciliter la reconnaissance après

``face_recognition_sface_2021dec.onnx``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Taille** : ~38.7 MB
- **Fonction** : Reconnaissance faciale
- **Sortie** : Vecteur de caractéristiques (embedding de 128 dimensions)

Téléchargement
~~~~~~~~~~~~~~

Les modèles sont téléchargés automatiquement depuis OpenCV Zoo lors de la première utilisation.

.. code-block:: bash

   # URLs de téléchargement
   YuNet:  https://github.com/opencv/opencv_zoo/.../face_detection_yunet_2023mar.onnx
   SFace:  https://github.com/opencv/opencv_zoo/.../face_recognition_sface_2021dec.onnx

Format des Encodages
--------------------

Vecteur de Caractéristiques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chaque visage est représenté par un vecteur de **128 nombres flottants** :

.. code-block:: python

   [0.123, -0.456, 0.789, ..., 0.321]  # 128 valeurs

Propriétés
~~~~~~~~~~

- **Dimension** : 128
- **Type** : ``numpy.ndarray`` de ``float32``
- **Normalisation** : Vecteur unitaire (norme L2 = 1)
