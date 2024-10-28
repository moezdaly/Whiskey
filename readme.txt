Die Arbeit im Whiskey-Projekt basiert ausschließlich auf der Google-Bibliothek TensorFlow und Keras.
Zunächst einmal muss ich erklären, dass ich nach einigen Tests am Anfang festgestellt habe, dass ein von Grund auf neu erstelltes Modell nichts im Vergleich zu Modellen ist, die auf bereits trainierten Modellen aufbauen.
Die Modelle, die auf einem bereits trainierten Modell aufbauen, zeigten extrem bessere Ergebnisse als die Modelle, die nicht trainiert wurden, also wurden die von Grund auf neu erstellten Modelle verworfen.
Die vorstehende Aussage lässt sich anhand der in img_classif_v1.ipynb gezeigten Ergebnisse belegen. Die Erklärung dazu findet sich in der Datei selbst in Form von Markdown-Texten über jeder Zelle sowie in ausführlichen Kommentaren.
Die betrachteten Modellbasen sind: normales Modell ohne vortrainierte Modelle, vgg16, inceptionv3, ResNet50, MobileNetV2 und EfficientNetB0


Die folgenden Dateien sind ähnlich aufgebaut und werden daher gemeinsam erläutert:
- computing_best_of_diff_lr_and_layers.ipynb
- computing_best_of_diff_lr_and_layers.py
- computing_big_models.py
- computing_big_models_16_32_32_32_32.py
- computing_big_models_v2.py
- computing_big_models_v3.py
- computing_big_models_v4_original.py

Sie beginnen mit dem Import der erforderlichen Bibliothek, laden die Trainings- und Testdaten (dieselben werden in allen Dateien in diesem Folder verwendet) und trainieren, kompilieren und testen dann verschiedene Modelle unter Verwendung verschiedener Basismodelle und verschiedener Schichtkomplexitäten und verschiedener Lernraten für die Strafbedingungen sowohl im Modell als auch im Compiler.
Für alle Dateien werden alle Ergebnisse in einer Textdatei namens output_log.txt gespeichert, da durch das Testen so vieler Modelle der Output sehr groß ist, was es unmöglich macht, alles in der IDE anzuzeigen.
Das beste vortrainierte Modell als Basis für unsere Modelle ist das inceptionv3-Modell. Mit diesem Modell als Grundlage habe ich verschiedene Kombinationen der oben genannten Parameter ausprobiert.
Die folgenden Modelle lieferten die besten Ergebnisse: (alle verwenden die Kreuzentropie als Verlustfunktion und die Sigmoidfunktion als Aktivierungsfunktion)
- Model with 5 layers having 16, 32, 32, 32, 32 nodes in each layer with no regulation terms.
- Model with 2 convolution 2d layers
- Model with layers as the follwing : 16, 32, 32, 32 with 0.01 regulization term in the training layers.
Starten Sie das Skript test_3.py, um die detaillierten Ergebnisse der genannten Modelle zu erhalten.