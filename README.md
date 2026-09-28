# Masterarbeit 
## Entwicklung eines Projected Augmented Reality Prototyps für den Einsatz bei neurochirurgischen Operationen

In diesem Repository befindet sich der Quellcode für die Erstellung des Prototyps.

Ziel ist es, Schnittmarken bei neurochirurgischen Operationen auf der Kopfhaut mittels Projektion anzuzeigen. 

Hierfür wird der Kopf in einem RGB-D-Bild mittels eines Point-Pair-Feature Algorithmus registriert und dann die Projektion erstellt.

> **Hinweis:** Dieses Repository verwendet Code von Julian Höpfinger sowie eine angepasste Version des PPF von [whateverforever/model-globally-match-locally-python](https://github.com/whateverforever/model-globally-match-locally-python) (eigener Fork: [EmmaSophieReichert/model-globally-match-locally-python](https://github.com/EmmaSophieReichert/model-globally-match-locally-python)).

## Installation und Anwendung

Klone das Repository inklusive der Submodule mit:
```bash
git clone --recurse-submodules [pfad zum Repository]
```

Lade den `pose_landmarker_full_task` von Mediapipe herunter und platziere ihn im Order `app/Tracking/FaceDetection`:
[https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)

Wenn die Submodule nicht geladen wurden, verwende:
```bash
git submodule update --init --recursive
```

Außerdem sollte für eine schnelle Ausführung des PPF-Alorithmus, die C++-Datei kompiliert werden. Verwende hierfür:
```bash
cd extern/ppf
mkdir build
cd build
cmake ..
make
```

Um die Daten vorzuverarbeiten, möchtest du vielleicht die Skripte im Ordner `scripts_before` ausführen, um zum Beispiel nur die äußere Hülle deines Meshs zu erhalten.

Stelle sicher, dass alle Requirements vorhanden sind und starte die Anwendung über:
```bash
python3 main.py
```

In der main.py lassen sich die verschiedenen Programmmodi auswählen. Für den normalen Gebrauch ist allerdings der bereits gewählte Standard-Modus ausreichend.
