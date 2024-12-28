# geRT Prototyp
# Import der erforfderlichen Bibliotheken:
import os # für Datei- und Verzeichnisoperationen
from transformers import AutoTokenizer, AutoModelForQuestionAnswering # NLP-Modelle und Tokenizer
from sentence_transformers import SentenceTransformer, util # semantic & similarity search
import torch # für Tensorberechnungen (Grundlage für NLP-Modelle)
from datetime import datetime # zum Speichern von Zeitstempeln für Konversationsprotokolle

# vortrainiertes Modell für semantische Textsuche
bert_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Eingabeaufforderung für das Verzeichnis mit den .txt-Dateien
directory_path = input("Geben Sie den Pfad zu Ihrem Verzeichnis mit den .txt-Dateien ein: ")

# Funktion zum Lesen und Laden des Korpus in ein Dictionary
def load_documents_from_directory(directory_path):
    # lädt alle .txt-Dateien aus dem angegebenen Verzeichnis und speichert den Inhalt in einem Dictionary
    # Schlüssel = Dateiname, Wert = Inhalt der Datei
    documents = {}
    for filename in os.listdir(directory_path): # Iteration über alle Dateien im Verzeichnis
        if filename.endswith('.txt'): # nur .txt-Dateien berücksichtigen
            file_path = os.path.join(directory_path, filename) # vollständigen Pfad zur Datei erstellen
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                # Datei lesen & im Dictionary speichern
                documents[filename] = file.read()
    return documents


# Funktion zur Suche nach dem relevantesten Dokument
def retrieve_documents(query, documents, model):
    # sucht das relevanteste Dokument zur User-Anfrage; berechnet semantische Ähnlichkeiten zwischen Anfrage &
    # Dokumentinhalten; gibt Dateinamen als Output zurück

    query_embedding = model.encode(query, convert_to_tensor=True)   # Embedding der Anfrage erzeugen
    document_embeddings = model.encode(list(documents.values()), convert_to_tensor=True)    # Embedding aller
    # Dokumente berechnen
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)     # Ähnlichkeiten zwischen der Anfrage
    # und den Dokumenten berechnen
    best_doc_index = torch.argmax(similarities).item()  # Index des ähnlichsten Dokuments finden
    best_document_name = list(documents.keys())[best_doc_index]     # Dateinamen des passendsten Dokuments ausgeben
    return best_document_name


# Funktion zur Extraktion relevanter Abschnitte
def extract_relevant_sections(query, document_content, model, num_sections=3, max_length=600):
    # Extrahiert relevanteste Abschnitte eines Dokuments basierend auf User-Anfrage
    # splittet das Dokument in Absätze; sortiert Absätze nach Relevanz basierend auf semantischer Ähnlichkeit
    # gibt top-N Abschnitte kombiniert zurück mit Beschränkung auf maximale Länge von 600 character
    # Dokument in Absätze teilen
    sections = document_content.split('\n\n')   # Dokument anhand von Doppelzeilenumbrüchen in Absätze teilen
    query_embedding = model.encode(query, convert_to_tensor=True)   # Embedding der Anfrage erzeugen
    section_embeddings = model.encode(sections, convert_to_tensor=True)     # Embeddings der Absätze berechnen
    similarities = util.pytorch_cos_sim(query_embedding, section_embeddings)    # Ähnlichkeit berechnen
    sorted_indices = torch.argsort(similarities, descending=True).flatten()     # Abschnitte nach Relevanz sortieren
    relevant_sections = [sections[i] for i in sorted_indices[:num_sections]]    # top-N Abschnitte auswählen
    combined_text = '\n\n'.join(relevant_sections)      # Abschnitte kombinieren
    if len(combined_text) > max_length:
        combined_text = combined_text[:max_length] + "..."  # Nach max. Länge Text Abschneiden und "..." anfügen
    return combined_text


# Funktion zum Speichern des Konversationsverlaufs in Textdatei mit Zeitstempel
def save_conversation(conversation_history):
    conversation_dir = "Konversationen"     # Verzeichnis für gespeicherte Konversationen
    if not os.path.exists(conversation_dir):    # Verzeichnis erstellen, falls noch nicht vorhanden
        os.makedirs(conversation_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # Zeitstempel für die Datei
    filename = f"conversation_{timestamp}.txt"      # Dateiname mit Zeitstempel
    filepath = os.path.join(conversation_dir, filename)     # Vollständiger Dateipfad
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(conversation_history)    # Konversation in die Datei schreiben

    print(f"Konversation gespeichert unter: {filepath}")


# Laden der Korpusdateien in ein Dictionary
documents = load_documents_from_directory(directory_path)

# Laden des Frage-Antwort-Modells für QA
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/gelectra-base-germanquad")    # Tokenizer
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/gelectra-base-germanquad")    # QA-Modell


# Funktion zur Antwortgenerierung auf Basis der User-Anfrage und des Kontexts
def generate_answer_with_qa(query, context):
    # Generiert eine Antwort auf die User-Anfrage, basierend auf dem Kontext aus dem Korpus
    # nutzt vortrainiertes Frage-Antwort-Modell (deepset/ gelectra-base-germanquad)
    inputs = qa_tokenizer.encode_plus(query, context, return_tensors="pt")  # Tokenisierung von Frage und Kontext
    answer_start_scores, answer_end_scores = qa_model(**inputs).values()     # Wahrscheinlichkeit für Start/
    # Ende der Antwort

    answer_start = torch.argmax(answer_start_scores)    # Startpunkt der Antwort
    answer_end = torch.argmax(answer_end_scores) + 1    # Endpunkt der Antwort
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])     # Antwort extrahieren
    )
    return answer

# Konversationsschleife für fortlaufende Fragen und Antworten:
conversation_history = ""   # speichert den Verlauf der Konversation
while True:
    # Dynamische Eingabe für die Anfrage durch den User
    query = input("Stellen Sie hier Ihre Frage an geRT (oder 'stop' zum Beenden): ")    # User-Anfrage
    if query.lower() == "stop":      # Abbruch der Konversation
        save_conversation(conversation_history)     # Verlauf speichern
        print("Konversation beendet.")
        break
    best_document_name = retrieve_documents(query, documents, bert_model)   # Relevantestes Dokument finden
    best_document_content = documents[best_document_name]   # Inhalt des Dokuments abrufen
    relevant_text = extract_relevant_sections(query, best_document_content, bert_model, num_sections=2, max_length=500)
    # relevante Abschnitte extrahieren

    print(f"Bestes Dokument: {best_document_name}")
    print(f"Relevante Abschnitte: {relevant_text}")     # Abschnitte anzeigen

    # Generiert die Antwort basierend auf dem Kontext und der Frage
    answer = generate_answer_with_qa(query, relevant_text)  # Antwort generieren

    print(f"geRT: {answer}")    # Antwort ausgeben
    conversation_history += f"Benutzer: {query}\ngeRT: {answer}\n\n"    # Verlauf aktualisieren
