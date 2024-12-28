import re
import os

def load_text(file_path):
    #Lädt den Text aus der angegebenen Datei
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file: #alternative Kodierung bei Fehlern mit UTF-8
            return file.read()

def save_text(text, file_path):
    #Speichert den bereinigten Text in einer neuen Datei
    cleaned_file_path = file_path.replace('.txt', '_bereinigt.txt')
    with open(cleaned_file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Die Bereinigung wurde abgeschlossen und der bereinigte Text wurde in '{cleaned_file_path}' gespeichert.")

def remove_unnecessary_special_characters(text):
    #Entfernt unnötige Sonderzeichen aus dem Text
    text = re.sub(r"[«*^]", "", text)  # Entfernt «, * und ^
    text = re.sub(r"[a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+", "", text)  # Entfernt alphanumerische Artefakte
    text = re.sub(r"mlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA", "", text)  # Entfernt bestimmte Zeichenfolge
    text = re.sub(r"ZYXWVUTSRQPONMLKJIHGFEDCBA", "", text)  # Entfernt Alphabet rückwärts
    text = re.sub(r"\s+", " ", text)  # Reduziert mehrfache Leerzeichen auf ein einziges
    text = re.sub(r"\s([?.!,:;])", r"\1", text)  # Entfernt Leerzeichen vor Satzzeichen
    return text

def correct_text_in_folder(folder_path):
    #Bereinigt alle .txt-Dateien im angegebenen Ordner von unnötigen Sonderzeichen
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') and not filename.endswith('_bereinigt.txt'):
            file_path = os.path.join(folder_path, filename)
            print(f"Verarbeite Datei: {file_path}")

            # Text laden, bereinigen und speichern
            text = load_text(file_path)
            text = remove_unnecessary_special_characters(text)
            save_text(text, file_path)

# Ordnerpfad angeben und den Bereinigungsprozess starten
folder_path = input("Pfad für Ordner mit Textdateien hier angeben: ")
correct_text_in_folder(folder_path)
