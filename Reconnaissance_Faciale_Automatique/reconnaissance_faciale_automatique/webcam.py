import cv2
import numpy as np
import os
import sys

# Add the project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from .manager import FaceRecognizerManager
except (ImportError, ValueError):
    from manager import FaceRecognizerManager

def main():
    print("Initialisation du gestionnaire de reconnaissance faciale...")
    # Initialize manager with the correct encoding file path used by the interface
    encoding_path = os.path.join(project_root, "encodings_data", "visages_connus.pkl")
    manager = FaceRecognizerManager(encoding_file=encoding_path)
    
    # Check and download models if needed
    if not manager.check_and_download_models(print):
        print("Erreur : Impossible de récupérer les modèles.")
        return

    # Load models
    if not manager.load_models():
        print("Erreur : Impossible de charger les modèles.")
        return
        
    # Load known encodings
    success, count = manager.load_encodings()
    if success:
        print(f"Chargé {count} visages connus.")
        # Mention loaded names
        unique_names = sorted(list(set(manager.known_names)))
        print(f"Personnes détectables : {', '.join(unique_names)}")
    else:
        print(f"Avertissement : Aucun visage connu chargé depuis {encoding_path}.")
        print("La reconnaissance ne fonctionnera pas (seulement détection).")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la webcam.")
        return

    print("Démarrage de la webcam. Appuyez sur 'q' pour quitter.")

    # Session tracking: keep track of who has been seen
    present_people = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire une image depuis la webcam.")
            break

        # Get frame dimensions
        h, w, _ = frame.shape
        
        if manager.detector is not None:
            # Detector requirements
            manager.detector.setInputSize((w, h))
            
            # Detect faces
            # faces output: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            _, faces = manager.detector.detect(frame)

            if faces is not None:
                for face in faces:
                    # bounding box
                    box = list(map(int, face[:4]))
                    x, y, w_box, h_box = box[0], box[1], box[2], box[3]
                    
                    # Recognition logic
                    name = "Inconnu"
                    color = (0, 0, 255) # Red for unknown
                    
                    if manager.known_features and manager.recognizer is not None:
                        # Align and extract features
                        face_align = manager.recognizer.alignCrop(frame, face)
                        if face_align is not None:
                            unknown_feat = manager.recognizer.feature(face_align)
                            
                            best_score = 0.0
                            best_name = "Inconnu"
                            
                            for i, known_feat in enumerate(manager.known_features):
                                score = manager.recognizer.match(known_feat, unknown_feat, cv2.FaceRecognizerSF_FR_COSINE)
                                if score > best_score:
                                    best_score = score
                                    if score > manager.threshold:
                                        best_name = manager.known_names[i]
                            
                            if best_name != "Inconnu":
                                name = f"{best_name}"
                                if best_name not in present_people:
                                    print(f"{best_name} est présent en cours")
                                    present_people.add(best_name)
                                
                                name = f"{best_name} ({best_score:.2f})"
                                color = (0, 255, 0) # Green for known

                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                    
                    # Draw name background
                    # Get text size
                    (text_w, text_h), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
                    
                    # Draw name
                    cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Reconnaissance Faciale - Webcam', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
