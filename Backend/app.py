from flask import Flask, request, jsonify
import spacy
import nltk
from nltk.corpus import wordnet as wn
from flask_cors import CORS

# Download WordNet data if not already downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Flask app and spaCy model
app = Flask(__name__)
CORS(app)
nlp = spacy.load("en_core_web_sm")

# Lesk algorithm implementation
def lesk_algorithm(sentence, target_word):
    # Tokenize the sentence using spaCy
    doc = nlp(sentence)
    
    # Find the target word in the sentence and get its POS
    target_token = None
    for token in doc:
        if token.text.lower() == target_word.lower():
            target_token = token
            break

    if target_token is None:
        return "Target word not found in the sentence."

    # Map spaCy POS tags to WordNet POS tags
    pos_mapping = {
        "NOUN": wn.NOUN,
        "VERB": wn.VERB,
        "ADJ": wn.ADJ,
        "ADV": wn.ADV
    }
    wordnet_pos = pos_mapping.get(target_token.pos_, wn.NOUN)  # Default to NOUN

    # Get all senses of the target word in WordNet
    senses = wn.synsets(target_word, pos=wordnet_pos)
    
    if not senses:
        return "No senses found for the target word in WordNet."

    # Lesk algorithm: find the sense with the most overlap with the context
    best_sense = None
    max_overlap = 0
    context = set([w.lemma_ for w in doc if w.is_alpha and not w.is_stop])
    
    for sense in senses:
        # Get definition and examples for each sense
        definition = set(sense.definition().lower().split())
        examples = set(word for example in sense.examples() for word in example.lower().split())
        
        # Calculate overlap between context and (definition + examples)
        overlap = len(context.intersection(definition.union(examples)))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    # Return the best sense's definition or a fallback message
    if best_sense:
        return best_sense.definition()
    else:
        return "Unable to determine the sense."

# API endpoint to predict word sense
@app.route('/predict', methods=['POST'])
def predict_sense():
    data = request.json
    sentence = data['sentence']
    target_word = data['target_word']

    # Use Lesk algorithm to predict sense
    sense = lesk_algorithm(sentence, target_word)

    # Return the sense as the response
    response = {"sense": sense}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
