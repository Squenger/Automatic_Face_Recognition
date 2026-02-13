Référence API
=============

Cette page documente l'API publique du module ``manager.py``, qui constitue le cœur du système de reconnaissance faciale.

.. contents:: Table des matières
   :local:
   :depth: 2

Vue d'ensemble
--------------

Le module ``manager.py`` fournit la classe principale ``FaceRecognizerManager`` qui gère l'ensemble du pipeline de reconnaissance faciale, de l'entraînement à l'identification.

Classe FaceRecognizerManager
-----------------------------

.. currentmodule:: reconnaissance_faciale_automatique.manager

.. autoclass:: FaceRecognizerManager
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Initialisation
~~~~~~~~~~~~~~

.. automethod:: FaceRecognizerManager.__init__
   :no-index:

Gestion des Modèles
~~~~~~~~~~~~~~~~~~~~

.. automethod:: FaceRecognizerManager.check_and_download_models
   :no-index:
.. automethod:: FaceRecognizerManager.load_models
   :no-index:

Gestion des Encodages
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: FaceRecognizerManager.load_encodings
   :no-index:

Entraînement
~~~~~~~~~~~~

.. automethod:: FaceRecognizerManager.train_faces
   :no-index:

Traitement d'Images
~~~~~~~~~~~~~~~~~~~

.. automethod:: FaceRecognizerManager.process_directory
   :no-index:

Gestion de la Base de Données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: FaceRecognizerManager.delete_person
   :no-index:
.. automethod:: FaceRecognizerManager.clear_database
   :no-index:

Exemple d'Utilisation
---------------------

Voici un exemple complet d'utilisation de la classe ``FaceRecognizerManager`` :

.. code-block:: python

   from reconnaissance_faciale_automatique.manager import FaceRecognizerManager

   # 1. Initialisation
   manager = FaceRecognizerManager(
       threshold=0.4  # Seuil de reconnaissance
   )

   # 2. Vérification et téléchargement des modèles
   def progress(msg):
       print(f"[INFO] {msg}")

   if manager.check_and_download_models(progress_callback=progress):
       print("✓ Modèles prêts")

   # 3. Chargement des modèles
   if manager.load_models():
       print("✓ Modèles chargés")

   # 4. Entraînement sur des visages connus
   success = manager.train_faces(
       known_dir="/path/to/known_faces",
       progress_callback=progress
   )

   if success:
       print("✓ Entraînement terminé")

   # 5. Chargement des encodages
   success, count = manager.load_encodings()
   print(f"✓ {count} visages chargés")

   # 6. Traitement d'images inconnues
   manager.process_directory(
       unknown_dir="/path/to/unknown_faces",
       progress_callback=progress
   )

   # 7. Consultation des résultats
   for filepath, names in manager.processed_images:
       print(f"{filepath}: {', '.join(names)}")

Gestion de la Base de Données
------------------------------

Suppression d'une personne
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Supprimer tous les encodages d'une personne
   success, count = manager.delete_person("Alice")
   if success:
       print(f"✓ {count} encodages supprimés")

Vidage complet
~~~~~~~~~~~~~~

.. code-block:: python

   # Vider complètement la base de données
   if manager.clear_database():
       print("✓ Base de données vidée")

Structure des Données
----------------------

Fichiers générés
~~~~~~~~~~~~~~~~

Le système génère deux fichiers principaux :

1. **visages_connus.pkl** : Fichier pickle contenant les paires (vecteurs de caractéristiques, noms)
2. **visages_connus_noms.txt** : Fichier texte avec la liste unique et triée des noms

Format des encodages
~~~~~~~~~~~~~~~~~~~~

Chaque visage est représenté par :

- Un vecteur de 128 dimensions (SFace embedding)
- Un nom associé (string)

Paramètres Importants
---------------------

Seuil de reconnaissance (threshold)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le paramètre ``threshold`` contrôle la sensibilité de la reconnaissance :

- **Valeurs basses (0.3-0.4)** : Plus permissif, risque de faux positifs
- **Valeurs moyennes (0.4-0.5)** : Équilibre recommandé
- **Valeurs hautes (0.5-0.6)** : Plus strict, risque de faux négatifs

Modèles ONNX
~~~~~~~~~~~~

Le système utilise deux modèles d'OpenCV Zoo :

- **YuNet** (face_detection_yunet_2023mar.onnx) : Détection de visages
- **SFace** (face_recognition_sface_2021dec.onnx) : Reconnaissance faciale

Callbacks de Progression
-------------------------

Toutes les méthodes de traitement acceptent un paramètre ``progress_callback`` optionnel :

.. code-block:: python

   def my_callback(message: str) -> None:
       """Fonction appelée pour chaque mise à jour de progression"""
       print(f"[PROGRESS] {message}")

   manager.train_faces(
       known_dir="/path/to/faces",
       progress_callback=my_callback
   )

Notes Techniques
----------------

- Les modèles sont téléchargés automatiquement depuis GitHub si absents
- L'entraînement est incrémental : seuls les nouveaux visages sont ajoutés
- Le renommage des fichiers gère automatiquement les collisions de noms
- Les images supportées : JPG, JPEG, PNG
- Un seul visage par image est encodé lors de l'entraînement (le premier détecté)
