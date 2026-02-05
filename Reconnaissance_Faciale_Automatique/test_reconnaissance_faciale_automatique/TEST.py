from unittest.mock import MagicMock, patch

import pytest

from reconnaissance_faciale_automatique.manager import FaceRecognizerManager


@pytest.fixture
def manager():
    """Fixture pour fournir une instance de FaceRecognizerManager."""
    return FaceRecognizerManager(
        model_dir="/tmp/models", encoding_file="/tmp/encodings.pkl"
    )


def test_manager_init(manager):
    """Teste l'initialisation de FaceRecognizerManager."""
    assert manager.model_dir == "/tmp/models"
    assert manager.encoding_file == "/tmp/encodings.pkl"
    assert manager.threshold == 0.4
    assert manager.detector is None
    assert manager.recognizer is None


@patch("os.path.exists")
@patch("os.makedirs")
@patch("urllib.request.urlretrieve")
def test_check_and_download_models(
    mock_urlretrieve, mock_makedirs, mock_exists, manager
):
    """Teste la vérification et le téléchargement des modèles."""
    # Les modèles n'existent pas
    mock_exists.return_value = False

    result = manager.check_and_download_models()

    assert result is True
    assert mock_makedirs.called
    assert mock_urlretrieve.call_count == 2


@patch("cv2.FaceDetectorYN.create")
@patch("cv2.FaceRecognizerSF.create")
def test_load_models(mock_sf_create, mock_yn_create, manager):
    """Teste le chargement des modèles ONNX."""
    mock_yn_create.return_value = MagicMock()
    mock_sf_create.return_value = MagicMock()

    result = manager.load_models()

    assert result is True
    assert manager.detector is not None
    assert manager.recognizer is not None


@patch("os.path.exists")
@patch("builtins.open")
@patch("pickle.load")
def test_load_encodings_success(mock_pickle_load, mock_open, mock_exists, manager):
    """Teste le chargement réussi des encodages."""
    mock_exists.return_value = True
    mock_pickle_load.return_value = (["feat1"], ["name1"])

    success, count = manager.load_encodings()

    assert success is True
    assert count == 1
    assert manager.known_names == ["name1"]


def test_load_encodings_not_found(manager):
    """Teste le chargement des encodages quand le fichier n'existe pas."""
    with patch("os.path.exists", return_value=False):
        success, count = manager.load_encodings()
        assert success is False
        assert count == 0


def test_rename_file_logic(manager):
    """Teste la logique de renommage de fichier sans appels OS réels."""
    with patch("os.path.exists") as mock_exists:
        with patch("os.rename") as mock_rename:
            # Renommage simple
            mock_exists.side_effect = [False]  # Le nouveau fichier n'existe pas
            result = manager._rename_file("/tmp", "image.jpg", {"Aimine"})
            assert result == "Aimine.jpg"
            mock_rename.assert_called_with("/tmp/image.jpg", "/tmp/Aimine.jpg")

            # Renommage avec plusieurs personnes
            mock_exists.side_effect = [False]
            result = manager._rename_file("/tmp", "image.jpg", {"Aimine", "Bob"})
            assert result == "Aimine_Bob.jpg"

            # Gestion des collisions
            # Premier essai : Aimine.jpg existe, Aimine_2.jpg n'existe pas
            mock_exists.side_effect = [True, False]
            result = manager._rename_file("/tmp", "image.jpg", {"Aimine"})
            assert result == "Aimine_2.jpg"
            mock_rename.assert_called_with("/tmp/image.jpg", "/tmp/Aimine_2.jpg")


@patch("os.path.exists")
@patch("os.path.isdir")
@patch("os.listdir")
@patch("cv2.imread")
def test_train_faces(mock_imread, mock_listdir, mock_isdir, mock_exists, manager):
    """Teste la logique du processus d'entraînement."""
    manager.detector = MagicMock()
    manager.recognizer = MagicMock()

    mock_exists.return_value = True
    mock_isdir.return_value = True  # Traiter "Alice" comme un répertoire
    # Structure: known_dir/Aimine/Léo.jpg
    mock_listdir.side_effect = [["Aimine"], ["Léo.jpg"]]

    mock_img = MagicMock()
    mock_img.shape = (100, 100, 3)
    mock_imread.return_value = mock_img

    # Mock de la détection et reconnaissance
    manager.detector.detect.return_value = (None, [MagicMock()])
    manager.recognizer.alignCrop.return_value = MagicMock()
    manager.recognizer.feature.return_value = "feature_vector"

    with patch("builtins.open", MagicMock()):
        with patch("pickle.dump") as mock_pickle_dump:
            result = manager.train_faces("/tmp/known")

            assert result is True
            assert manager.known_names == ["Aimine"]
            assert manager.known_features == ["feature_vector"]
            assert mock_pickle_dump.called


@patch("os.path.exists")
@patch("os.listdir")
@patch("cv2.imread")
def test_process_directory(mock_imread, mock_listdir, mock_exists, manager):
    """Teste le traitement d'un répertoire de visages inconnus."""
    manager.known_features = ["known_feat"]
    manager.known_names = ["Aimine"]
    manager.detector = MagicMock()
    manager.recognizer = MagicMock()

    mock_exists.return_value = True
    mock_listdir.return_value = ["unknown.jpg"]

    mock_img = MagicMock()
    mock_img.shape = (100, 100, 3)
    mock_imread.return_value = mock_img

    # Mock de détection : un visage trouvé
    manager.detector.detect.return_value = (None, [MagicMock()])
    manager.recognizer.alignCrop.return_value = MagicMock()
    manager.recognizer.feature.return_value = "unknown_feat"

    # Mock de reconnaissance : score > seuil
    manager.recognizer.match.return_value = 0.9
    manager.threshold = 0.4

    with patch.object(manager, "_rename_file") as mock_rename:
        mock_rename.return_value = "Aimine.jpg"
        manager.process_directory("/tmp/unknown")

        assert mock_rename.called
        # Vérifier qu'il a passé le bon ensemble de noms trouvés
        args, _ = mock_rename.call_args
        assert args[2] == {"Aimine"}


@patch("os.path.exists")
@patch("urllib.request.urlretrieve")
def test_check_and_download_models_download_error(
    mock_urlretrieve, mock_exists, manager
):
    """Teste le téléchargement de modèles quand le téléchargement échoue."""
    callback_messages = []

    def callback(msg):
        callback_messages.append(msg)

    # Premier appel pour model_dir, puis pour chaque fichier de modèle
    mock_exists.side_effect = [False, False, False]
    mock_urlretrieve.side_effect = Exception("Network error")

    with patch("os.makedirs"):
        result = manager.check_and_download_models(progress_callback=callback)

        assert result is False
        assert any("Erreur de téléchargement" in msg for msg in callback_messages)


@patch("cv2.FaceDetectorYN.create")
def test_load_models_failure(mock_yn_create, manager):
    """Teste le chargement des modèles quand une exception se produit."""
    mock_yn_create.side_effect = Exception("Model loading error")

    result = manager.load_models()

    assert result is False
    assert manager.detector is None


@patch("os.path.exists")
@patch("builtins.open")
@patch("pickle.load")
def test_load_encodings_error(mock_pickle_load, mock_open, mock_exists, manager):
    """Teste le chargement des encodages quand pickle.load échoue."""
    mock_exists.return_value = True
    mock_pickle_load.side_effect = Exception("Corrupted file")

    success, count = manager.load_encodings()

    assert success is False
    assert count == 0


def test_save_database_creates_directory(manager):
    """Teste que _save_database crée le répertoire s'il n'existe pas."""
    manager.known_features = ["feat1"]
    manager.known_names = ["Alice"]

    with patch("os.path.exists", return_value=False):
        with patch("os.makedirs") as mock_makedirs:
            with patch("builtins.open", MagicMock()):
                with patch("pickle.dump"):
                    manager._save_database()
                    assert mock_makedirs.called


def test_delete_person_success(manager):
    """Teste la suppression d'une personne de la base de données."""
    manager.known_features = ["feat1", "feat2", "feat3"]
    manager.known_names = ["Alice", "Bob", "Alice"]

    with patch.object(manager, "_save_database"):
        success, count = manager.delete_person("Alice")

        assert success is True
        assert count == 2
        assert manager.known_names == ["Bob"]
        assert len(manager.known_features) == 1


def test_delete_person_empty_database(manager):
    """Teste la suppression depuis une base de données vide."""
    manager.known_features = []
    manager.known_names = []

    success, count = manager.delete_person("Alice")

    assert success is False
    assert count == 0


def test_clear_database(manager):
    """Teste le vidage complet de la base de données."""
    manager.known_features = ["feat1", "feat2"]
    manager.known_names = ["Alice", "Bob"]

    with patch("os.path.exists", return_value=True):
        with patch("os.remove") as mock_remove:
            result = manager.clear_database()

            assert result is True
            assert manager.known_features == []
            assert manager.known_names == []
            assert mock_remove.call_count == 2  # Fichiers pkl et txt


def test_train_faces_directory_not_found(manager):
    """Teste l'entraînement quand le répertoire n'existe pas."""
    callback_messages = []

    def callback(msg):
        callback_messages.append(msg)

    manager.detector = MagicMock()
    manager.recognizer = MagicMock()

    with patch("os.path.exists", return_value=False):
        result = manager.train_faces("/nonexistent", progress_callback=callback)

        assert result is False
        assert any("introuvable" in msg for msg in callback_messages)


def test_train_faces_skip_unreadable_images(manager):
    """Teste que l'entraînement ignore les images qui ne peuvent pas être lues."""
    manager.detector = MagicMock()
    manager.recognizer = MagicMock()

    with patch("os.path.exists", return_value=True):
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir") as mock_listdir:
                mock_listdir.side_effect = [["Alice"], ["bad.jpg", "good.jpg"]]
                with patch("cv2.imread") as mock_imread:
                    # La première image échoue au chargement, la seconde réussit
                    mock_imread.side_effect = [None, MagicMock(shape=(100, 100, 3))]
                    manager.detector.detect.return_value = (None, [MagicMock()])
                    manager.recognizer.alignCrop.return_value = MagicMock()
                    manager.recognizer.feature.return_value = "feat"

                    with patch("builtins.open", MagicMock()):
                        with patch("pickle.dump"):
                            result = manager.train_faces("/tmp/known")

                            assert result is True
                            assert len(manager.known_features) == 1


def test_process_directory_no_encodings(manager):
    """Teste le traitement quand aucun encodage n'est chargé."""
    callback_messages = []

    def callback(msg):
        callback_messages.append(msg)

    manager.known_features = []

    manager.process_directory("/tmp/unknown", progress_callback=callback)

    assert any("Aucune signature chargée" in msg for msg in callback_messages)
