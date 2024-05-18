import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def noise_removal(text):
    # Remove non-alphabetic characters and extra whitespaces
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def character_normalization(text):
    # Convert characters to lowercase
    return text.lower()

def text_processing_pipeline(text):
    # Step 0: Noise removal
    cleaned_text = noise_removal(text)
    
    # Step 1: Character normalization
    normalized_text = character_normalization(cleaned_text)

    # Step 2: Tokenization
    tokens = word_tokenize(normalized_text)
    # Remove stop words (Step 0 continued)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    
    
    # Step 3: Part-of-speech tagging
    pos_tags = pos_tag(tokens)
    
    # Step 4: Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = " ".join([lemmatizer.lemmatize(word) for word in tokens])

    # Step 5: Named entity recognition
    named_entities = ne_chunk(pos_tags)
    
    # Print the results
    print("Original Text:", text)
    print("\nCleaned Text (Noise Removal and Character Normalization):", cleaned_text)
    print("\n1. Tokenization:")
    print(tokens)
    
    print("\n2. Part-of-Speech Tagging:")
    print(pos_tags)

    print("\n3. Lemmatization:")
    print(lemmatized_text)

    print("\n4. Named Entity Recognition:")
    print(named_entities)

if __name__ == "__main__":
    # Example text
    text = "Demonstrate that the claims made from the analysis are warranted and have produced findings with methodological integrity. The procedures that support methodological integrity typically are described across the relevant sections of a paper, but they could be addressed in a separate section when elaboration or emphasis would be helpful. Issues of methodological integrity include the following: Assess the adequacy of the data in terms of its ability to capture forms of diversity most relevant to the question, research goals, and inquiry approach."
    # Run the text processing pipeline
    text_processing_pipeline(text)

