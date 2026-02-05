# Reconnaissance Faciale Automatique



## Requirements

- Python >= 3.9
- OpenCV
- PyQt6
- NumPy

### How to run

Launch the application by running:

```bash
$ uv run main_qt
# OR
$ python -m main_qt
```

You can also use the following scripts to launch the application on macOS:
- `Lancer_Interface.command`: Launches the GUI.


## Data structure

In order the application works correctly, images must be organized was following :


### 1. Known Faces (Training)
The `known_faces/` folder must contain one subfolder per person. The folder name will be used as the identifier for recognition.
```text
known_faces/
├── Aimine_Meddeb/
│   ├── photo1.jpg
│   └── photo2.png
├── Leonard_Beddouk/
│   ├── image_01.jpg
│   └── selfie.jpeg
```

### 2. Folder to Sort
The `unknown_faces/` folder (or any other target folder chosen in the interface) contains the photos you want to automatically identify and rename.
```text
unknown_faces/
├── vacances_2023_001.jpg
├── DSC_0982.png
└── random_img.jpeg
```

### 3. Database (Automatic)
The application generates an `encodings_data/` folder at the root to store the calculated facial signatures:
- `visages_connus.pkl`: Contains the feature vectors.
- `visages_connus_noms.txt`: Simple list of registered names.

# Workflow

1. **Verify models** : Click "1. Vérifier Modèles" to download required models
2. **Training** : Click "2. Apprendre Visages" to train on known faces
3. **Processing** : Click "3. Lancer le Tri" to process and rename unknown images
4. **Visualization** : Click "4. Voir les Résultats" to view processed images with:

## Development

### How to run pre-commit

```bash
uvx pre-commit run -a
```

Alternatively, you can install it so that it runs before every commit :

```bash
uvx pre-commit install
```

### How to run tests

```bash
uv sync --group Reconnaissance_Faciale_Automatique/test_reconnaissance_faciale_automatique
uv run coverage run -m pytest -v
```
### COVERAGE

```bash
uv run coverage report
```

### How to run type checking

```bash
uvx pyright reconnaissance_faciale_automatique --pythonpath .venv/bin/python
```

### How to build docs

```bash
uv sync
cd docs && uv run sphinx-build . _build
```

### How to run autobuild for docs 

```bash
uv sync
cd docs && uv run sphinx-autobuild --re-ignore generated --host 0.0.0.0 --watch ".." . _build
```

http://localhost:8000 
