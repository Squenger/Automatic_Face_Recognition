"""This module shows you how you can construct a nice documentation with
sphinx and the right syntaxe for docstrings.
"""

import os
import pickle
import urllib.request
from collections.abc import Callable

import cv2


class FaceRecognizerManager:
    """
    Gère la détection et la reconnaissance faciale via les modèles ONNX d'OpenCV Zoo.
    Cette classe permet l'entraînement (encodage) et l'identification automatique
    d'images dans un répertoire donné.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        encoding_file: str | None = None,
        threshold: float = 0.4,
    ) -> None:
        r"""Initialise le gestionnaire de reconnaissance faciale avec support pour détection et reconnaissance d'objets.

        Cette méthode configure le gestionnaire en initialisant les chemins des modèles ONNX,
        le fichier de stockage des encodages et le seuil de similarité pour la reconnaissance faciale.
        Les chemins par défaut sont définis de manière robuste pour fonctionner dans l'environnement du projet.

        Notes:
            - Les modèles ONNX utilisent YuNet pour la détection et SFace pour la reconnaissance
            - Le seuil de similarité par défaut (0.4) peut être ajusté selon la précision souhaitée
            - Les chemins par défaut pointent vers des répertoires standards du projet

        Args:
            model_dir: Chemin du répertoire contenant les modèles ONNX.
                Si ``None``, utilise le dossier ``models_onnx`` du package.
                Type: :obj:`str` ou ``None``
            encoding_file: Chemin du fichier pickle stockant les encodages de visages connus.
                Si ``None``, utilise ``encodings_data/visages_connus.pkl`` à la racine du projet.
                Type: :obj:`str` ou ``None``
            threshold: Seuil de similarité cosinus pour valider une correspondance de visage.
                Les valeurs proches de 1.0 sont plus strictes. Plage recommandée: [0.3, 0.6].
                Type: :obj:`float`

        Returns:
            ``None``. L'objet est initialisé et prêt pour l'entraînement et la reconnaissance.
        """
        # Détermination du dossier racine du projet (Projet PAI)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )

        if model_dir is None:
            # Chemin par défaut vers le dossier des modèles dans le package
            model_dir = os.path.join(os.path.dirname(__file__), "models_onnx")

        if encoding_file is None:
            # Chemin par défaut robuste dans Projet PAI/encodings_data
            encoding_file = os.path.join(
                project_root, "encodings_data", "visages_connus.pkl"
            )

        self.model_dir = model_dir
        self.encoding_file = encoding_file
        self.threshold = threshold

        self.detector = None
        self.recognizer = None

        self.known_features = []
        self.known_names = []

        # Pour stocker les résultats du traitement (chemin, noms reconnus)
        self.processed_images = []

        # URLs des modèles provenant d'OpenCV Zoo
        self.models_files = {
            "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?raw=true",
            "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?raw=true",
        }

    def check_and_download_models(
        self, progress_callback: Callable[[str], None] | None = None
    ) -> bool:
        r"""Vérifie la présence des modèles ONNX et les télécharge depuis OpenCV Zoo si nécessaire.

        Cette méthode contrôle l'existence des fichiers de modèles (YuNet et SFace) dans le répertoire
        configuré. Si un modèle est manquant, la méthode procède à son téléchargement automatique depuis
        les dépôts GitHub officiels d'OpenCV Zoo.

        Notes:
            - Les modèles sont volumineux (~100 MB au total), le téléchargement peut être long
            - Une connexion Internet est requise pour le téléchargement
            - Les fichiers téléchargés sont conservés dans ``model_dir`` pour un usage futur
            - La fonction de rappel est appelée avec des messages de progression

        Args:
            progress_callback: Fonction optionnelle recevant des messages de statut en chaîne de caractères.
                Appelée périodiquement pour informer de la vérification et du téléchargement.
                Type: :obj:`callable` acceptant un :obj:`str`, ou ``None``

        Returns:
            ``True`` si tous les modèles sont disponibles et téléchargés avec succès,
            ``False`` en cas d'erreur de téléchargement ou d'accès au répertoire.
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if progress_callback:
            progress_callback("Vérification des modèles...")

        for filename, url in self.models_files.items():
            filepath = os.path.join(self.model_dir, filename)
            if not os.path.exists(filepath):
                if progress_callback:
                    progress_callback(f"Téléchargement de {filename}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Erreur de téléchargement {filename}: {e}")
                    return False

        if progress_callback:
            progress_callback("Tous les modèles sont opérationnels.")
        return True

    def load_models(self) -> bool:
        r"""Initialise les modèles de détection et reconnaissance faciale à partir des fichiers ONNX.

        Crée les instances OpenCV des modèles YuNet (pour la détection de visages) et SFace
        (pour l'extraction de caractéristiques et la reconnaissance). Ces instances sont stockées
        comme attributs d'instance pour une utilisation répétée sans rechargement.

        Notes:
            - YuNet utilise une taille d'entrée de 320×320 pixels, ajustée dynamiquement au traitement
            - Le seuil de score pour YuNet est fixé à 0.8 pour une détection fiable
            - SFace extrait des vecteurs de 128 dimensions pour chaque visage détecté
            - Cette méthode doit être appelée avant ``train_faces()`` ou ``process_directory()``

        Args:
            Aucun paramètre.

        Returns:
            ``True`` si le chargement des deux modèles réussit sans exception,
            ``False`` si une erreur survient (fichiers manquants, incompatibilité ONNX, etc.).
        """
        try:
            # Création du détecteur de visages YuNet
            self.detector = cv2.FaceDetectorYN.create(
                model=os.path.join(self.model_dir, "face_detection_yunet_2023mar.onnx"),
                config="",
                input_size=(320, 320),  # Ajusté dynamiquement lors du traitement
                score_threshold=0.8,
                nms_threshold=0.3,
                top_k=5000,
            )

            # Création du reconnaisseur SFace
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=os.path.join(
                    self.model_dir, "face_recognition_sface_2021dec.onnx"
                ),
                config="",
            )
            return True
        except Exception as e:
            print(f"Erreur lors de l'initialisation des modèles : {e}")
            return False

    def load_encodings(self) -> tuple[bool, int]:
        r"""Charge les encodages de visages connus depuis le fichier pickle de la base de données.

        Lit le fichier spécifié lors de l'initialisation contenant les paires (vecteur de caractéristiques, nom).
        Ces données sont stockées dans les attributs ``known_features`` et ``known_names`` pour une utilisation
        dans la reconnaissance faciale. Gère automatiquement les cas où le fichier n'existe pas.

        Notes:
            - Le fichier doit contenir un tuple (features_list, names_list) via pickle
            - Les chemins absolus sont affichés dans les messages de log pour le débogage
            - Cette méthode est appelée automatiquement par ``train_faces()`` pour un apprentissage incrémental
            - En cas d'absence de fichier, une liste vide est conservée

        Args:
            Aucun paramètre.

        Returns:
            Un tuple ``(success, count)`` où:

            - ``success`` (:obj:`bool`): ``True`` si le fichier a pu être lu sans erreur, ``False`` sinon
            - ``count`` (:obj:`int`): Nombre de visages chargés (0 si le fichier n'existe pas ou erreur)
        """
        abs_path = os.path.abspath(self.encoding_file)
        if os.path.exists(self.encoding_file):
            try:
                with open(self.encoding_file, "rb") as f:
                    self.known_features, self.known_names = pickle.load(f)
                print(
                    f"Chargement réussi depuis : {abs_path} ({len(self.known_names)} visages)"
                )
                return True, len(self.known_names)
            except Exception as e:
                print(f"Erreur lors du chargement de {abs_path} : {e}")
                return False, 0
        print(f"Fichier de base de données introuvable : {abs_path}")
        return False, 0

    def _save_database(self) -> tuple[str | None, str | None]:
        r"""Sauvegarde les encodages faciales dans deux fichiers: pickle et texte.

        Persiste les données d'entraînement (vecteurs de caractéristiques et noms associés) dans:
        - Un fichier pickle contenant les paires (features, names) pour la reconnaissance
        - Un fichier texte contenant la liste unique et triée des noms pour consultation

        Crée automatiquement le répertoire de destination s'il n'existe pas.

        Notes:
            - Fichier pickle: ``encoding_file`` (format: pickle binaire)
            - Fichier texte: ``encoding_file`` avec extension ``.txt`` au lieu de ``.pkl``
            - Les noms du fichier texte sont uniques et triés alphabétiquement
            - Méthode protégée (préfixe ``_``), usage interne recommandé
            - Les fichiers existants sont écrasés silencieusement

        Args:
            Aucun paramètre. Utilise les attributs ``self.known_features``,
            ``self.known_names`` et ``self.encoding_file``.

        Returns:
            Tuple ``(pkl_path, txt_path)`` avec:

            - ``pkl_path`` (:obj:`str`): Chemin absolu du fichier pickle (ou ``None`` si ``encoding_file`` non défini)
            - ``txt_path`` (:obj:`str`): Chemin du fichier texte contenant les noms (ou ``None`` si erreur)
        """
        if not self.encoding_file:
            return None, None

        save_dir = os.path.dirname(self.encoding_file)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Sauvegarde Pickle (Signatures + Noms)
        abs_path = os.path.abspath(self.encoding_file)
        with open(self.encoding_file, "wb") as f:
            pickle.dump((self.known_features, self.known_names), f)

        # Sauvegarde Texte (Liste des noms unique)
        names_file = self.encoding_file.replace(".pkl", "_noms.txt")
        unique_names = sorted(list(set(self.known_names)))
        with open(names_file, "w", encoding="utf-8") as f:
            f.write("\n".join(unique_names))

        return abs_path, names_file

    def train_faces(
        self, known_dir: str, progress_callback: Callable[[str], None] | None = None
    ) -> bool:
        r"""Entraîne le système en encodant les visages connus depuis une structure de répertoires hiérarchique.

        Explore un répertoire contenant des sous-dossiers (un par personne) remplies d'images de visages.
        Extrait les vecteurs de caractéristiques de chaque visage détecté et les associe au nom du dossier.
        Supporte l'apprentissage incrémental en ajoutant uniquement les nouvelles personnes à la base existante.

        Notes:
            - Structure attendue: ``known_dir/Personne1/*.jpg``, ``known_dir/Personne2/*.jpg``, etc.
            - Formats d'image supportés: JPG, JPEG, PNG
            - Un seul visage par image est encodé (le premier détecté)
            - Les modèles sont chargés automatiquement si nécessaire
            - L'apprentissage incrémental évite de réentrainer les visages existants
            - Une sauvegarde est effectuée automatiquement à la fin

        Args:
            known_dir: Chemin du répertoire contenant les sous-dossiers de personnes.
                Type: :obj:`str`
            progress_callback: Fonction de rappel recevant des messages de progression en chaîne.
                Type: :obj:`callable` acceptant un :obj:`str`, ou ``None``

        Returns:
            ``True`` si l'entraînement réussit et les données sont sauvegardées,
            ``False`` si le répertoire n'existe pas, aucun visage n'est détecté, ou erreur de chargement des modèles.
        """
        if not self.detector or not self.recognizer:
            if not self.load_models():
                return False

        # Charger les visages existants pour un apprentissage incrémental
        self.load_encodings()
        existing_names = set(self.known_names)

        if not os.path.exists(known_dir):
            if progress_callback:
                progress_callback(f"Erreur : Le dossier {known_dir} est introuvable.")
            return False

        people_dirs = [
            d
            for d in os.listdir(known_dir)
            if os.path.isdir(os.path.join(known_dir, d))
        ]
        # Filtrer pour ne garder que les nouveaux visages
        new_people = [d for d in people_dirs if d not in existing_names]
        total_new = len(new_people)

        if total_new == 0:
            msg = "Aucun nouveau visage à apprendre. La base est déjà à jour."
            if progress_callback:
                progress_callback(msg)
            print(msg)
            return True

        for idx, name in enumerate(new_people):
            dir_path = os.path.join(known_dir, name)
            if progress_callback:
                progress_callback(f"Analyse de : {name} ({idx + 1}/{total_new})")

            images_processed = 0
            images_skipped = 0
            faces_found = 0

            for filename in os.listdir(dir_path):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                filepath = os.path.join(dir_path, filename)
                img = cv2.imread(filepath)
                if img is None:
                    images_skipped += 1
                    if progress_callback:
                        progress_callback(f"  ⚠️  Image non lisible : {filename}")
                    continue

                images_processed += 1

                # Détection faciale
                h, w, _ = img.shape
                if self.detector is not None:
                    self.detector.setInputSize((w, h))
                    _, faces = self.detector.detect(img)

                    if faces is not None and len(faces) > 0:
                        # Alignement et extraction des caractéristiques (features)
                        if self.recognizer is not None:
                            face_align = self.recognizer.alignCrop(img, faces[0])
                            face_feature = self.recognizer.feature(face_align)

                            self.known_features.append(face_feature)
                            self.known_names.append(name)
                            faces_found += 1
                            if progress_callback:
                                progress_callback(f"  ✓ Visage détecté dans : {filename}")
                    else:
                        if progress_callback:
                            progress_callback(
                                f"  ✗ Aucun visage détecté dans : {filename}"
                            )

            # Résumé pour cette personne
            if progress_callback:
                progress_callback(
                    f"  → {name} : {faces_found} visage(s) sur {images_processed} image(s) traitée(s)"
                )

        if not self.known_features:
            msg = "ERREUR : Aucun visage n'a été trouvé dans le dossier. La sauvegarde est annulée pour éviter d'effacer la base existante."
            if progress_callback:
                progress_callback(msg)
            print(msg)
            return False

        abs_path, names_file = self._save_database()

        display_names_file = os.path.basename(names_file) if names_file else "inconnu"
        msg_fin = f"Entraînement terminé. {len(self.known_features)} signatures sauvegardées dans {abs_path}.\nListe des noms sauvegardée dans : {display_names_file}"
        if progress_callback:
            progress_callback(msg_fin)
        print(msg_fin)
        return True

    def delete_person(self, name_to_delete: str) -> tuple[bool, int]:
        r"""Supprime tous les encodages d'une personne de la base de données.

        Parcourt la liste des noms connus et supprime toutes les lignes correspondant à la personne.
        Les données restantes sont automatiquement sauvegardées dans les fichiers pickle et texte.

        Notes:
            - La recherche est sensible à la casse (case-sensitive)
            - Une sauvegarde est effectuée automatiquement si au moins une signature est supprimée
            - Si la personne n'existe pas, aucune modification n'est effectuée
            - Les fichiers de sauvegarde sont mis à jour après suppression

        Args:
            name_to_delete: Nom exact de la personne à supprimer de la base.
                Doit correspondre exactement aux noms stockés.
                Type: :obj:`str`

        Returns:
            Tuple ``(success, count)`` avec:

            - ``success`` (:obj:`bool`): ``True`` si au moins une signature a été supprimée, ``False`` sinon
            - ``count`` (:obj:`int`): Nombre de signatures supprimées pour la personne
        """
        if not self.known_names:
            return False, 0

        indices_to_keep = [
            i for i, name in enumerate(self.known_names) if name != name_to_delete
        ]
        num_deleted = len(self.known_names) - len(indices_to_keep)

        if num_deleted > 0:
            self.known_features = [self.known_features[i] for i in indices_to_keep]
            self.known_names = [self.known_names[i] for i in indices_to_keep]
            self._save_database()
            return True, num_deleted

        return False, 0

    def clear_database(self) -> bool:
        r"""Vide complètement la base de données et supprime les fichiers de sauvegarde.

        Réinitialise les attributs ``known_features`` et ``known_names`` à des listes vides,
        puis supprime les fichiers pickle et texte associés du disque. Cette opération est irréversible.

        Notes:
            - Supprime le fichier pickle contenant les encodages
            - Supprime le fichier texte contenant la liste des noms
            - Les fichiers qui n'existent pas sont ignorés silencieusement
            - Idéal pour réinitialiser complètement le système de reconnaissance
            - Opération définitive: les données ne peuvent pas être récupérées

        Args:
            Aucun paramètre.

        Returns:
            ``True`` si l'opération réussit (les fichiers sont supprimés ou n'existent pas).
        """
        self.known_features = []
        self.known_names = []

        # Supprimer les fichiers s'ils existent
        if os.path.exists(self.encoding_file):
            os.remove(self.encoding_file)

        names_file = self.encoding_file.replace(".pkl", "_noms.txt")
        if os.path.exists(names_file):
            os.remove(names_file)

        return True

    def process_directory(
        self, unknown_dir: str, progress_callback: Callable[[str], None] | None = None
    ) -> None:
        r"""Traite toutes les images d'un répertoire, identifie les personnes et renomme automatiquement les fichiers.

        Analyse chaque image du répertoire, détecte les visages, extrait leurs caractéristiques,
        les compare à la base d'encodages connus et renomme les fichiers avec les noms des personnes identifiées.
        Les résultats (chemin final, noms identifiés) sont stockés dans ``self.processed_images``.

        Notes:
            - Formats supportés: JPG, JPEG, PNG (vérification par extension)
            - Les fichiers sans visages détectés sont ignorés
            - Le seuil de similarité (``self.threshold``) détermine si un visage est reconnu
            - Nom de fichier: ``Personne1_Personne2.jpg`` si plusieurs visages sont détectés
            - Numérotation automatique en cas de collision (``Personne_2.jpg``)
            - La méthode requiert une base d'encodages chargée (``load_encodings()``)

        Args:
            unknown_dir: Chemin du répertoire contenant les images à traiter.
                Type: :obj:`str`
            progress_callback: Fonction optionnelle recevant des messages de progression en chaîne.
                Appelée tous les 5 fichiers traités pour éviter les surcharges.
                Type: :obj:`callable` acceptant un :obj:`str`, ou ``None``

        Returns:
            ``None``. Les résultats sont stockés dans ``self.processed_images``
            (list de tuples (chemin_final, liste_noms)).
        """
        if not self.known_features:
            if progress_callback:
                progress_callback(
                    "Erreur : Aucune signature chargée. Lancez l'entraînement d'abord."
                )
            return

        if not os.path.exists(unknown_dir):
            if progress_callback:
                progress_callback(f"Dossier introuvable : {unknown_dir}")
            return

        # Réinitialiser la liste des images traitées
        self.processed_images = []

        files = [
            f
            for f in os.listdir(unknown_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        total_files = len(files)
        renamed_count = 0

        for idx, filename in enumerate(files):
            filepath = os.path.join(unknown_dir, filename)

            if progress_callback and idx % 5 == 0:
                progress_callback(f"Traitement en cours : {idx + 1}/{total_files}...")

            img = cv2.imread(filepath)
            if img is None:
                continue

            # Mise à jour de la taille d'entrée pour le détecteur
            h, w, _ = img.shape
            found_names_in_image = set()

            if self.detector is not None:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(img)

                if faces is not None and len(faces) > 0:
                    for face in faces:
                        if self.recognizer is not None:
                            face_align = self.recognizer.alignCrop(img, face)
                            unknown_feat = self.recognizer.feature(face_align)

                            best_score = 0.0
                            best_name = "Inconnu"

                            # Comparaison avec les signatures connues
                            for i, known_feat in enumerate(self.known_features):
                                score = self.recognizer.match(
                                    known_feat,
                                    unknown_feat,
                                    cv2.FaceRecognizerSF_FR_COSINE,
                                )

                                if score > best_score:
                                    best_score = score
                                    if score > self.threshold:
                                        best_name = self.known_names[i]

                            if best_name != "Inconnu":
                                found_names_in_image.add(best_name)

            # Renommage du fichier si des visages sont identifiés
            new_filepath = filepath  # Par défaut, le fichier n'est pas renommé
            if found_names_in_image:
                new_name = self._rename_file(
                    unknown_dir, filename, found_names_in_image
                )
                if new_name:
                    renamed_count += 1
                    new_filepath = os.path.join(unknown_dir, new_name)
                    if progress_callback:
                        progress_callback(f"Renommé : {filename} -> {new_name}")

            # Stocker le résultat (chemin final, noms reconnus)
            sorted_names = (
                sorted(list(found_names_in_image))
                if found_names_in_image
                else ["Inconnu"]
            )
            self.processed_images.append((new_filepath, sorted_names))

        if progress_callback:
            progress_callback(
                f"Traitement terminé. {renamed_count} images identifiées sur {total_files}."
            )

    def _rename_file(
        self, directory: str, filename: str, found_names: set[str]
    ) -> str | None:
        r"""Renomme un fichier image avec les noms des personnes identifiées, en gérant les collisions.

        Construit un nouveau nom de fichier basé sur les noms détectés, joignant les noms triés
        par des underscores. Gère les collisions en ajoutant un suffixe numérique (``_2``, ``_3``, etc.)
        si le nom cible existe déjà. Renomme effectivement le fichier sur le disque.

        Notes:
            - Les noms sont triés alphabétiquement avant d'être jointe
            - L'extension d'origine du fichier est conservée
            - Méthode protégée (préfixe ``_``), usage interne recommandé
            - En cas d'erreur lors du renommage du système, retourne ``None`` sans exception
            - Le fichier n'est renommé que s'il y a un nouveau nom différent

        Args:
            directory: Chemin absolu du répertoire contenant le fichier.
                Type: :obj:`str`
            filename: Nom d'origine du fichier (sans le chemin).
                Type: :obj:`str`
            found_names: Ensemble (set) des noms de personnes identifiées dans l'image.
                Type: :obj:`set[str]`

        Returns:
            ``str``: Nouveau nom du fichier si le renommage réussit,
            ``None`` si le fichier n'a pas pu être renommé (erreur OSError) ou si aucune modification n'était nécessaire.
        """
        sorted_names = sorted(list(found_names))
        new_base_name = "_".join(sorted_names)

        _, ext = os.path.splitext(filename)
        new_filename = f"{new_base_name}{ext}"
        filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)

        # Gestion des collisions (ex: personne_2.jpg)
        counter = 2
        final_new_filename = new_filename
        final_new_filepath = new_filepath

        while os.path.exists(final_new_filepath) and final_new_filepath != filepath:
            final_new_filename = f"{new_base_name}_{counter}{ext}"
            final_new_filepath = os.path.join(directory, final_new_filename)
            counter += 1

        if filepath != final_new_filepath:
            try:
                os.rename(filepath, final_new_filepath)
                return final_new_filename
            except OSError:
                return None
        return None
