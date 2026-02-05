# Architecture du Projet - Reconnaissance Faciale Automatique



## Fichiers Principaux

### 1. `reconnaissance_faciale_automatique/manager.py`

**Rôle** : Cœur du système de reconnaissance faciale. Gère toute la logique de détection, reconnaissance, entraînement et traitement des images.

**Classe principale** : `FaceRecognizerManager`

**Fonctionnalités** :

#### Initialisation et Configuration
- `__init__()` : Initialise le gestionnaire avec les chemins des modèles, fichier d'encodage et seuil de reconnaissance
- Configure les attributs pour les modèles ONNX (détecteur et reconnaisseur)
- Initialise les listes pour stocker les caractéristiques et noms connus

#### Gestion des Modèles
- `check_and_download_models()` : Vérifie la présence des modèles ONNX et les télécharge depuis OpenCV Zoo si nécessaire
  - YuNet : Modèle de détection de visages
  - SFace : Modèle de reconnaissance faciale
- `load_models()` : Charge les modèles ONNX dans OpenCV (FaceDetectorYN et FaceRecognizerSF)

#### Gestion de la Base de Données
- `load_encodings()` : Charge les encodages de visages depuis le fichier pickle
  - Retourne le succès et le nombre de visages chargés
- `_save_database()` : Sauvegarde les encodages dans deux fichiers :
  - `.pkl` : Format pickle pour les vecteurs de caractéristiques
  - `.txt` : Format texte pour la liste des noms
- `delete_person()` : Supprime tous les encodages d'une personne spécifique
- `clear_database()` : Vide complètement la base de données

#### Entraînement
- `train_faces()` : Entraîne le système sur un répertoire de visages connus
  - Parcourt les sous-dossiers (un par personne)
  - Détecte les visages dans chaque image
  - Extrait les vecteurs de caractéristiques avec SFace
  - Sauvegarde les encodages dans la base de données

#### Traitement et Reconnaissance
- `process_directory()` : Traite un répertoire d'images inconnues
  - Détecte les visages dans chaque image
  - Compare avec la base de données
  - Renomme automatiquement les fichiers avec les noms identifiés
  - Retourne la liste des images traitées avec les noms reconnus
- `_rename_file()` : Gère le renommage intelligent des fichiers
  - Gère les collisions de noms (ajoute _2, _3, etc.)
  - Supporte plusieurs personnes dans une même image


---

### 2. `reconnaissance_faciale_automatique/main_qt.py`

**Rôle** : Interface graphique principale de l'application utilisant PyQt6.

**Classes principales** :

#### `FaceRecoApp` (Fenêtre principale)
**Fonctionnalités** :
- Interface utilisateur complète avec 4 étapes principales :
  1. Vérification et téléchargement des modèles
  2. Entraînement sur les visages connus
  3. Traitement et tri des images inconnues
  4. Visualisation des résultats

**Méthodes principales** :
- `init_ui()` : Construit l'interface graphique
  - Sélecteurs de répertoires pour visages connus et inconnus
  - Slider pour ajuster le seuil de reconnaissance
  - Boutons d'action pour chaque étape
  - Zone de log pour les messages
  - Boutons de gestion de la base de données
- `create_file_input()` : Crée des widgets de sélection de fichiers
- `browse_folder()` : Ouvre un dialogue de sélection de dossier
- `log_message()` : Affiche des messages dans la zone de log
- `update_threshold()` : Met à jour le seuil de reconnaissance
- `toggle_buttons()` : Active/désactive les boutons pendant le traitement

**Gestion des tâches asynchrones** :
- `start_worker()` : Lance un thread de travail pour éviter de bloquer l'interface
- `run_check_models()` : Vérifie les modèles en arrière-plan
- `run_training()` : Lance l'entraînement en arrière-plan
- `run_processing()` : Traite les images en arrière-plan
- `on_processing_finished()` : Callback appelé à la fin du traitement

**Gestion de la base de données** :
- `delete_person_ui()` : Interface pour supprimer une personne
- `clear_database_ui()` : Interface pour vider la base de données

**Visualisation** :
- `show_results()` : Affiche la fenêtre de visualisation des résultats
- `apply_styles()` : Applique les styles CSS personnalisés

#### `ImageViewerWindow` (Visionneuse d'images)
**Fonctionnalités** :
- Affiche les images traitées avec navigation
- Montre les noms reconnus pour chaque image
- Navigation avec boutons Précédent/Suivant
- Redimensionnement automatique des images

**Méthodes** :
- `init_ui()` : Construit l'interface de visualisation
- `load_image()` : Charge et affiche l'image courante
- `show_previous()` / `show_next()` : Navigation entre les images
- `resizeEvent()` : Gère le redimensionnement de la fenêtre

#### `WorkerThread` (Thread de travail)
**Fonctionnalités** :
- Exécute les tâches lourdes en arrière-plan
- Émet des signaux pour communiquer avec l'interface
- Évite le gel de l'interface pendant les opérations longues

**Fonction d'entrée** :
- `run()` : Point d'entrée pour lancer l'application

**Corrections macOS** :
- Gestion spéciale des plugins Qt pour éviter les conflits avec Anaconda
- Configuration de `QT_PLUGIN_PATH` et `QT_QPA_PLATFORM_PLUGIN_PATH`

---

### 3. `reconnaissance_faciale_automatique/webcam.py`

**Rôle** : Mode webcam en temps réel pour la reconnaissance faciale.

**Fonction principale** : `main()`

**Fonctionnalités** :
- Initialise le gestionnaire de reconnaissance
- Télécharge et charge les modèles si nécessaire
- Charge les encodages de visages connus
- Ouvre la webcam (device 0)
- Boucle de traitement en temps réel :
  - Capture des frames de la webcam
  - Détection des visages avec YuNet
  - Reconnaissance avec SFace
  - Affichage des rectangles et noms sur les visages
  - Suivi des personnes présentes (affiche un message lors de la première détection)

**Affichage** :
- Rectangle vert pour les personnes reconnues
- Rectangle rouge pour les inconnus
- Nom et score de confiance affichés
- Touche 'q' pour quitter


---

## Fichiers de Tests

### `test_reconnaissance_faciale_automatique/TEST.py`

**Rôle** : Suite de tests unitaires utilisant pytest pour valider le fonctionnement du gestionnaire.

**Couverture** : **18 tests** avec **86% de couverture** sur `manager.py`

**Documentation complète** : Voir [`TESTS.md`](TESTS.md) pour tous les détails

#### Tests de Fonctionnalités de Base (8 tests)

1. **`test_manager_init`** : Vérifie l'initialisation correcte du gestionnaire
2. **`test_check_and_download_models`** : Teste le téléchargement des modèles (mocké)
3. **`test_load_models`** : Vérifie le chargement des modèles ONNX (mocké)
4. **`test_load_encodings_success`** : Teste le chargement réussi des encodages
5. **`test_load_encodings_not_found`** : Teste le cas où le fichier d'encodage n'existe pas
6. **`test_rename_file_logic`** : Vérifie la logique de renommage des fichiers
   - Renommage simple
   - Renommage avec plusieurs personnes
   - Gestion des collisions
7. **`test_train_faces`** : Teste le processus d'entraînement (mocké)
8. **`test_process_directory`** : Teste le traitement d'un répertoire (mocké)

#### Tests de Gestion d'Erreurs et Cas Limites 

9. **`test_check_and_download_models_download_error`** : Gestion des erreurs réseau lors du téléchargement
10. **`test_load_models_failure`** : Gestion des exceptions lors du chargement des modèles
11. **`test_load_encodings_error`** : Gestion des fichiers pickle corrompus
12. **`test_save_database_creates_directory`** : Création automatique de répertoires
13. **`test_delete_person_success`** : Suppression d'une personne de la base
14. **`test_delete_person_empty_database`** : Suppression depuis une base vide
15. **`test_clear_database`** : Vidage complet de la base de données
16. **`test_train_faces_directory_not_found`** : Gestion des répertoires manquants
17. **`test_train_faces_skip_unreadable_images`** : Ignore les images corrompues
18. **`test_process_directory_no_encodings`** : Traitement sans encodages chargés

**Fixture** :
- `manager()` : Fournit une instance de `FaceRecognizerManager` pour les tests

**Exécution** :
```bash
uv sync
uv run coverage run -m pytest -v
uv run coverage report  # Voir la couverture
```

---

## Fichiers de Configuration

### `pyproject.toml`

**Rôle** : Configuration centrale du projet Python (format PEP 518).

**Sections** :

#### `[project]`
- Nom : `PAI_2025_Leonard_Aimine_Reconnaissance_faciale_automatiseeV2`
- Version : `0.1.0`
- Auteurs : Aimine Meddeb, Leonard Beddouk
- Python requis : `>=3.10, <3.14`
- Dépendances principales :
  - `pyside6` : Framework Qt (alternative)
  - `PyQt6` : Framework Qt pour l'interface graphique
  - `opencv-python` : Bibliothèque de vision par ordinateur

#### `[project.scripts]`
- `main_qt` : Point d'entrée pour l'interface graphique
- `webcam` : Point d'entrée pour le mode webcam

#### `[dependency-groups]`
- **test** : pytest, coverage, pytest-sugar, pandas-stubs
- **docs** : Sphinx, pydata-sphinx-theme, myst-parser, nbsphinx, etc.

#### `[tool.pytest.ini_options]`
- Configuration des tests
- Chemins de test : `test_reconnaissance_faciale_automatique` et `manager.py`
- Options : doctest activé, mode import importlib

#### `[tool.coverage.run]`
- Source de couverture : `reconnaissance_faciale_automatique`

#### `[build-system]`
- Backend : hatchling
- Inclut uniquement le package `reconnaissance_faciale_automatique`

---

### `.gitignore`

**Rôle** : Spécifie les fichiers et répertoires à ignorer par Git.

**Catégories** :
- Fichiers Python compilés : `*.py[cod]`, `__pycache__`
- Environnements virtuels : `.venv`
- Fichiers de build : `_build`, `dist/`, `target/`
- IDE : `.idea/`, `.vim`
- Système : `.DS_Store`, `Thumbs.db`
- Notebooks : `.ipynb_checkpoints/`
- Médias volumineux : `*.mp4`, `*.avi`, etc.
- Spécifique au projet : `jobs_dir/`

---

### `.pre-commit-config.yaml`

**Rôle** : Configuration des hooks pre-commit pour maintenir la qualité du code.

**Hooks configurés** :

#### `pre-commit-hooks` (v5.0.0)
- `trailing-whitespace` : Supprime les espaces en fin de ligne
- `end-of-file-fixer` : Ajoute une ligne vide en fin de fichier
- `check-merge-conflict` : Détecte les marqueurs de conflit de merge
- `check-case-conflict` : Détecte les conflits de casse de noms de fichiers
- `check-json` : Valide la syntaxe JSON
- `check-toml` : Valide la syntaxe TOML
- `check-ast` : Vérifie la syntaxe Python
- `debug-statements` : Détecte les instructions de débogage
- `check-yaml` : Valide la syntaxe YAML

#### `ruff-pre-commit` (v0.12.0)
- `ruff-check` : Linter Python moderne
  - Correction automatique activée
  - Target : Python 3.10+
  - Extensions : UP (pyupgrade), I (isort)
- `ruff-format` : Formateur de code Python

**Utilisation** :
```bash
uvx pre-commit run -a          # Exécuter manuellement
uvx pre-commit install         # Installer pour tous les commits
```

---

## Scripts de Lancement

### `Lancer_Interface.command`

**Rôle** : Script bash pour lancer l'interface graphique sur macOS.

**Contenu** :
```bash
#!/bin/bash
cd "$(dirname "$0")"
uv run main_qt
```

**Fonctionnement** :
1. Change le répertoire vers celui du script
2. Lance l'interface graphique via uv

**Utilisation** : Double-clic sur le fichier dans Finder

---

### `Lancer_Webcam.command`

**Rôle** : Script bash pour lancer le mode webcam sur macOS.

**Contenu** :
```bash
#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Webcam Recognition..."
uv run webcam
```

**Fonctionnement** :
1. Change le répertoire vers celui du script
2. Affiche un message de démarrage
3. Lance le mode webcam via uv

**Utilisation** : Double-clic sur le fichier dans Finder

---

## Documentation

### `docs/conf.py`

**Rôle** : Configuration Sphinx pour générer la documentation HTML.

**Configuration** :

#### Projet
- Nom : "Reconnaissance Faciale Automatique"
- Auteurs : Aimine Meddeb, Leonard Beddouk
- Copyright : 2025

#### Thème
- `pydata_sphinx_theme` : Thème moderne et responsive
- Logo : SupOptique
- Liens externes : Site de l'Institut d'Optique
- Lien GitHub vers le dépôt

#### Extensions Sphinx
- `sphinx.ext.autodoc` : Documentation automatique depuis les docstrings
- `sphinx.ext.napoleon` : Support des docstrings Google/NumPy
- `sphinx.ext.autosummary` : Génération automatique de résumés
- `sphinx.ext.intersphinx` : Liens vers d'autres documentations
- `sphinx.ext.viewcode` : Liens vers le code source
- `sphinx.ext.mathjax` : Support des formules mathématiques
- `myst_parser` : Support du Markdown
- `nbsphinx` : Intégration des notebooks Jupyter
- `sphinx_copybutton` : Bouton de copie pour les blocs de code
- `sphinx_favicon` : Gestion des favicons
- `sphinxarg.ext` : Documentation des arguments CLI

#### Options
- `autodoc_typehints = "signature"` : Affiche les types dans les signatures
- `nbsphinx_execute = "auto"` : Exécution automatique des notebooks
- `copybutton_prompt_text` : Détection intelligente des prompts à exclure

#### Intersphinx
- Liens vers la documentation Python, pandas, numpy

**Construction** :
```bash
cd docs && uv run make html        # Build une fois
cd docs && make livehtml           # Build avec auto-refresh
```

---

### `docs/index.rst`

**Rôle** : Page d'accueil de la documentation Sphinx.

**Structure** :
- Table des matières (toctree)
- Liens vers les différentes sections de la documentation
- Référence au README
- Documentation du module `reconnaissance_faciale_automatique`
- Exemples dans les notebooks

---


## Modèles ONNX

### `reconnaissance_faciale_automatique/models_onnx/`

**Contenu** :

#### `face_detection_yunet_2023mar.onnx` (232 KB)
- **Fonction** : Détection de visages
- **Modèle** : YuNet (mars 2023)
- **Source** : OpenCV Zoo
- **Sortie** : Coordonnées des boîtes englobantes et points de repère faciaux
- **URL** : https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

#### `face_recognition_sface_2021dec.onnx` (38.7 MB)
- **Fonction** : Reconnaissance faciale
- **Modèle** : SFace (décembre 2021)
- **Source** : OpenCV Zoo
- **Sortie** : Vecteurs de caractéristiques (embeddings) pour la comparaison
- **URL** : https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

**Téléchargement automatique** : Ces modèles sont téléchargés automatiquement par `manager.check_and_download_models()` s'ils ne sont pas présents.

---

## Base de Données

### `encodings_data/` (généré automatiquement à chaque apprentisage)

Cela évite d'avoir à stocker les vecteurs de caractéristiques dans le code source.

**Contenu** :

#### `visages_connus.pkl`
- **Format** : Pickle Python (pour le traitement par le programe)
- **Contenu** : Tuple de deux listes
  - Liste 1 : Vecteurs de caractéristiques (numpy arrays)
  - Liste 2 : Noms correspondants (strings)
- **Utilisation** : Chargé en mémoire pour la reconnaissance
- **Génération** : Créé par `manager.train_faces()`

#### `visages_connus_noms.txt`
- **Format** : Texte brut (un nom par ligne)
- **Contenu** : Liste simple des noms enregistrés
- **Utilisation** : Référence rapide pour vérifier les personnes détectables par un humain
- **Génération** : Créé en même temps que le fichier pickle

---

## Configuration GitHub

### `.github/workflows/`

**Rôle** : Workflows CI/CD pour l'automatisation.

**Workflows typiques** :
- Test unitaire (pytest)
- Build de la documentation (sphinx)
- Vérification des types (pyright)
- Vérification du formatage du code (pre-commit)

### `.github/ISSUE_TEMPLATE/`

**Rôle** : Templates pour standardiser les issues GitHub.

**Templates typiques** :
- Bug report
- Feature request

### `.github/pull_request_template.md`

**Rôle** : Template pour les pull requests.

**Contenu ** :
- Code style follows black conventions
- Type hints are consistent
- New lines are covered by tests
- If documentation is added, it does not raise a warning in sphinx

---



