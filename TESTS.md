# Documentation des Tests

## Résumé de la Couverture des Tests au 05/02/2026

- **Total de Tests** : 18
- **Couverture sur `manager.py`** : **86%**
- **Tous les Tests Réussis** : 18/18


## Catégories de Tests

### Tests de Fonctionnalités de Base (8 tests)

#### 1. `test_manager_init`
**Objectif** : Teste l'initialisation de FaceRecognizerManager  
**Valide** :
- Chemin correct du répertoire des modèles
- Chemin correct du fichier d'encodage
- Valeur par défaut du seuil (0.4)
- Le détecteur et le reconnaisseur sont initialement None

#### 2. `test_check_and_download_models`
**Objectif** : Teste la vérification et le téléchargement des modèles  
**Valide** :
- Création du répertoire quand les modèles n'existent pas
- Les fichiers de modèles sont téléchargés (2 fichiers : YuNet + SFace)
- Retourne True en cas de succès

#### 3. `test_load_models`
**Objectif** : Teste le chargement des modèles ONNX  
**Valide** :
- FaceDetectorYN est créé avec succès
- FaceRecognizerSF est créé avec succès
- Le détecteur et le reconnaisseur ne sont pas None après chargement

#### 4. `test_load_encodings_success`
**Objectif** : Teste le chargement réussi des encodages  
**Valide** :
- Les encodages sont chargés depuis le fichier pickle
- Nombre correct d'encodages chargés
- Les noms sont correctement assignés

#### 5. `test_load_encodings_not_found`
**Objectif** : Teste le chargement des encodages quand le fichier n'existe pas  
**Valide** :
- Retourne False quand le fichier n'existe pas
- Le compteur est à 0

#### 6. `test_rename_file_logic`
**Objectif** : Teste la logique de renommage de fichier sans appels OS réels  
**Valide** :
- Renommage simple avec une seule personne
- Renommage avec plusieurs personnes (noms concaténés)
- Gestion des collisions (ajoute _2, _3, etc.)

#### 7. `test_train_faces`
**Objectif** : Teste la logique du processus d'entraînement  
**Valide** :
- Parcours des répertoires
- Détection de visages dans les images
- Extraction des caractéristiques
- Sauvegarde de la base de données

#### 8. `test_process_directory`
**Objectif** : Teste le traitement d'un répertoire de visages inconnus  
**Valide** :
- Détection de visages dans les images inconnues
- Reconnaissance faciale contre la base de données connue
- Renommage de fichiers basé sur les résultats de reconnaissance

---

### Tests de Gestion d'Erreurs et Cas Limites (10 tests)

#### 9. `test_check_and_download_models_download_error`
**Objectif** : Teste le téléchargement de modèles quand le téléchargement échoue  
**Valide** :
- Gestion des erreurs réseau
- Message d'erreur approprié via callback
- Retourne False en cas d'échec

#### 10. `test_load_models_failure`
**Objectif** : Teste le chargement des modèles quand une exception se produit  
**Valide** :
- Gestion des exceptions pendant le chargement des modèles
- Le détecteur reste None en cas d'échec
- Retourne False

#### 11. `test_load_encodings_error`
**Objectif** : Teste le chargement des encodages quand pickle.load échoue  
**Valide** :
- Gestion des fichiers corrompus
- Retourne False en cas d'erreur
- Le compteur est à 0

#### 12. `test_save_database_creates_directory`
**Objectif** : Teste que _save_database crée le répertoire s'il n'existe pas  
**Valide** :
- Création du répertoire quand il est manquant
- La base de données est sauvegardée avec succès

#### 13. `test_delete_person_success`
**Objectif** : Teste la suppression d'une personne de la base de données  
**Valide** :
- La personne est retirée de known_names
- Les caractéristiques correspondantes sont supprimées
- Retourne le nombre correct d'entrées supprimées
- La base de données est sauvegardée après suppression

#### 14. `test_delete_person_empty_database`
**Objectif** : Teste la suppression depuis une base de données vide  
**Valide** :
- Retourne False quand la base de données est vide
- Le compteur est à 0
- Aucune erreur ne se produit

#### 15. `test_clear_database`
**Objectif** : Teste le vidage complet de la base de données  
**Valide** :
- Toutes les caractéristiques sont effacées
- Tous les noms sont effacés
- Les fichiers .pkl et .txt sont supprimés
- Retourne True en cas de succès

#### 16. `test_train_faces_directory_not_found`
**Objectif** : Teste l'entraînement quand le répertoire n'existe pas  
**Valide** :
- Message d'erreur approprié via callback
- Retourne False
- Aucun crash ne se produit

#### 17. `test_train_faces_skip_unreadable_images`
**Objectif** : Teste que l'entraînement ignore les images qui ne peuvent pas être lues  
**Valide** :
- Les images corrompues/invalides sont ignorées
- L'entraînement continue avec les images valides
- Retourne True si au moins un visage est trouvé

#### 18. `test_process_directory_no_encodings`
**Objectif** : Teste le traitement quand aucun encodage n'est chargé  
**Valide** :
- Message d'avertissement approprié via callback
- Aucun crash ne se produit
- Gestion gracieuse d'une base de données vide

---

## Détails de la Couverture

### Lignes Couvertes : 191/221 (86%)

### Lignes Non Couvertes (30 lignes)

Les 14% de lignes non couvertes restantes sont principalement :
- **Messages de callback de progression** (lignes 153, 370-374, 379, 417, 560-562, 579, 628, 639)
- **Instructions de journalisation** (lignes 383, 406-410, 583)
- **Calculs de chemins** (lignes 71, 75)
- **Conditions de cas limites** (lignes 288, 348-349, 467, 704-706)

Ces lignes sont difficiles à tester sans complexité excessive et représentent des chemins de code non critiques (journalisation, retour d'information UI).

---

## Métriques de Qualité des Tests

- **Aucun test instable** : Tous les tests sont déterministes
- **Exécution rapide** : ~0.3 secondes au total
- **Isolés** : Chaque test est indépendant
- **Dépendances mockées** : Aucune E/S de fichier réelle ni chargement de modèle
- **Assertions claires** : Chaque test a des assertions spécifiques et significatives
- **Documentation en français** : Toutes les docstrings en français pour la cohérence

---

## Exécution de Tests Spécifiques

### Exécuter un seul test
```bash
uv run pytest test_reconnaissance_faciale_automatique/TEST.py::test_manager_init -v
```

### Exécuter les tests correspondant à un motif
```bash
uv run pytest test_reconnaissance_faciale_automatique/TEST.py -k "delete" -v
```

### Exécuter avec sortie détaillée
```bash
uv run pytest test_reconnaissance_faciale_automatique/TEST.py -vv
```

---

## Intégration Continue

Les tests sont conçus pour s'exécuter dans les pipelines CI/CD :
- Aucune dépendance externe requise
- Toutes les dépendances sont mockées
- Exécution rapide
- Résultats déterministes
