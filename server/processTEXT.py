
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import traceback


from spacy import *


import os
import re




import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt



from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from transformers import pipeline
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tag import pos_tag 
import nltk

import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer


import pandas as pd

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer


# download nltk corpus (first time only)
import nltk

PATH= os.path.dirname(__file__)
debug = False

    
    

def read_file(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    subjects = data.split("\n, ")
    return [parse_subject(subject) for subject in subjects if subject]

def parse_subject(subject):
    if ("\n; ")in subject:
        sections = subject.split("\n; ")
        section_arr = [parse_section(section) for section in sections[1:] if section]
        return{sections[0]: section_arr}

    else:
        section_arr = [parse_section(subject)]
        return {"":section_arr}

def parse_section(section):
    if ("\n. ")in section:
        subsections = section.split("\n. ")
        subsection_arr = [parse_subsection(subsection) for subsection in subsections[1:] if subsection]
        return {subsections[0]: subsection_arr}
    else:
        subsection_arr = [parse_subsection(section)]
        return {"":subsection_arr}
         
    

def parse_subsection(subsection):
    if ("\n: ")in subsection:
        points = subsection.split("\n: ")
        point_arr = [parse_point(point) for point in points[1:] if point]

        return {points[0]: point_arr}
    else:
        point_arr = [parse_point(subsection)]
        return {"":point_arr}

def parse_point(point):
    subpoints = point.split("\n- ")
    subpoint_arr = [parse_subpoint(subpoint) for subpoint in subpoints[1:] if subpoint]
    return {subpoints[0]: subpoint_arr}

    
    
def parse_subpoint(subpoint):

    references = subpoint.split("\n> ")
    reference_arr = [reference for reference in references[1:] if reference]
    return {references[0]: reference_arr}


def qatest(question,context):
    qa_model = pipeline("question-answering")
    print(qa_model(question = question, context = context))
    


#model = "https://tfhub.dev/google/nnlm-en-dim50/2"
#hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
#hub_layer(train_examples[:3])

#Google's Universal Sentence Encoder: https://tfhub.dev/google/universal-sentence-encoder/2

def is_subset(subset, superset):
    return subset.issubset(superset)


def compareold(rule,text):
    nlp=load("en_use_md")
    ruletoken=nlp(rule)
    texttoken=nlp(text)
    if debug:
        print(ruletoken.similarity(texttoken))
    return ruletoken.similarity(texttoken)

def extract_author_names(text):
    sentences = sent_tokenize(text)
    author_names = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word for word in words if word.isalpha()]  # Remove punctuation
        words = [word for word in words if word.lower() not in stopwords.words('english')]  # Remove stopwords

        tagged_words = pos_tag(words)
        
        author_candidate = ""
        for word, tag in tagged_words:
            if tag == 'NNP':  # Proper noun
                author_candidate += word + " "
            elif author_candidate:
                author_names.append(author_candidate.strip())
                author_candidate = ""

    return author_names

def sense(text):
    nlp = load("en_core_web_sm")
    doc = nlp("text")
    for sent in doc.sents:
        if sent[0].is_title and sent[-1].is_punct:
            has_noun = 2
            has_verb = 1
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                    has_noun -= 1
                elif token.pos_ == "VERB":
                    has_verb -= 1
            if has_noun < 1 and has_verb < 1:
                if debug:
                    print(" makes sense"+ text)
                return 1


#def processtext(textarr):
 #   for text in textarr:

#def findsamplesize(text):
    

    
def analyse(text):
    
    nlp = load('en_core_web_sm')
    # Load the English language model for spaCy
    # Process the text with spaCy
    proctext = nlp(text)
    #for token in proctext:
     #   print(token.text, token.has_vector, token.vector_norm, token.is_oov)
    # Identify the named entities in the text
    entities = [ent.text for ent in proctext.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']]
    # Identify the main variables and theoretical issues under investigation and the relationships between them
    # Here is a simple example that looks for any tokens tagged as nouns or noun phrases
    variables = [token.text for token in proctext if token.pos_ in ['NOUN', 'NOUN_CHUNK']]

    # Print the results of the analysis
    if debug:
        print("Named entities:", entities)
        print("Main variables and theoretical issues:", variables)        

    return entities,variables


def classifyabstraction(text):
    classifier = pipeline("zero-shot-classification")
    candidate_labels = ["abstract", "concrete"]
    res = classifier(text, candidate_labels)
    if debug:
        for item in res:
            print('>>> ', item['sequence'], f'The label was {item["labels"][0]} with {round(item["scores"][0], 2)} percent confidence\n\n')
    return res

def comparebasic(text,rule):

    # Load the English language model for spaCy
    nlp = load('en_core_web_sm')
    # Process the text with spaCy
    proctext = nlp(text)
    # Process the rule with spaCy
    procrule = nlp(rule)
    # Compare the text and the rule
    similarity = proctext.similarity(procrule)
    if debug:
        print("Similarity:", similarity)
    return similarity

def vectorize(documents):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    return vectors, vectorizer

def find_paragraph_topic(paragraph):
    sentences = sent_tokenize(paragraph)
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence)]
    stop_words = set(stopwords.words('english'))
    
    ignore_terms=[line for line in open(os.path.join(PATH,'sectionimg\\ignoreterms.txt'))]
    filtered_words = [word for word in words if word not in stop_words and word not in ignore_terms]
    
   
    pos_tags = pos_tag(filtered_words)
    

    nouns = [word for word, pos in pos_tags if pos.startswith('N')]

    noun_freq = Counter(nouns)  
    topic = noun_freq.most_common(3)
    return topic

def remstoplem(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    filtered_words = [lemmatizer.lemmatize(word.lower()) for word in text if not word in stop_words]
    return stop_words,lemmatizer,filtered_words





def extract_entities(text):
    # Tokenize the paragraph into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    

    # Remove stop words and lemmatize the words
    sent_stop_words, sent_lemmatizer, sent_filtered_words = remstoplem(sentences)
    word_stop_words, word_lemmatizer, word_filtered_words = remstoplem(words)
    # Identify the named entities in the paragraph
    sententities = nltk.chunk.ne_chunk(nltk.pos_tag(sent_filtered_words))
    wordentities = nltk.chunk.ne_chunk(nltk.pos_tag(word_filtered_words))
    
    
    if debug:
        #print(filtered_words)
        #print(stop_words)
        #print(lemmatizer)
        print(sententities)
        #print(wordentities)
    # Extract the relevant information from the named entities
    sent_entity_names = []
    word_entity_names = []
    for entity in sententities:
        if hasattr(entity, 'label') and entity.label() == 'NE':
            sent_entity_names.append(' '.join(c[0] for c in entity))
    
    for wentity in wordentities:
        if hasattr(wentity, 'label') and wentity.label() == 'NE':
            word_entity_names.append(' '.join(c[0] for c in wentity))


    return sent_entity_names,word_entity_names
    

#horrific
def extract_sentences(arr):
    sentences = []
    for subject in arr:
        for subject_title, sectionsarr in subject.items():
            for sections in sectionsarr :
                for section_title, subsectionsarr in sections.items():
                    for subsections in subsectionsarr:
                        for subsection_title, pointarr in subsections.items():
                            for point in pointarr:
                                for point_title, subpointarr in point.items():
                                    if subpointarr:
                                        for subpoint in subpointarr:
                                            for subpoint_title, referencearr in subpoint.items():
                                                if referencearr:
                                                    for reference in referencearr:
                                                        sentences.append([subject_title, section_title, subsection_title, point_title+" "+ subpoint_title+" "+ reference])
                                                else:
                                                    sentences.append([subject_title, section_title, subsection_title, point_title+" "+ subpoint_title])
                                    else:
                                        sentences.append([subject_title, section_title, subsection_title, point_title])
    return sentences
 

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
from sklearn.metrics import accuracy_score

def find_matching_sentences(text, rule):
    sentences = preprocess(text)
    
    expected_content = ["abstract","introduction", "methodology", "results", "conclusion"]
    sectionlocationdict={}
    for content in expected_content:
        found = False
        for i,tokens in enumerate(sentences):
            if content in tokens:
                found = True
                sectionlocationdict[content]=i
                break
        if found:
            print(f"'{content}' found in the document.")
        else:
            print(f"'{content}' not found in the document.")
            
    matching_sentences = []
    
    # Iterate over each sentence in the sentences list
    for sentence in sentences:
        # Use re.search() to find matches of the rule in the sentence
        print(sentence)
        if re.search(r'\b' + rule + r'\b', sentence, re.IGNORECASE):
            # If a match is found, add the sentence to the matching_sentences list
            matching_sentences.append(sentence)
    return matching_sentences

def calculate_accuracy(guidelines, text):
    accuracy_per_guideline = {}
    for guideline in guidelines:
        matching_sentences = find_matching_sentences(text, guideline)
        guideline_found = len(matching_sentences) > 0
        accuracy_per_guideline[guideline] = guideline_found, matching_sentences
        
    accuracy = sum([guideline_found for guideline_found, _ in accuracy_per_guideline.values()]) / len(guidelines)

    return accuracy, accuracy_per_guideline





def preprocess(text):
    # Tokenize into sentences and preprocess
    preprocessed_sentences = [sentence.lower() for sentence in text]
    tokenized_lines = [word_tokenize(line) for line in preprocessed_sentences]
    print(tokenized_lines)
    return tokenized_lines

def calculate_similarity(target_info, sections):
    # Preprocess target information and sections
    target_info_preprocessed = preprocess(target_info)
    sections_preprocessed = [preprocess(section) for section in sections]
    print(target_info_preprocessed)
    print(sections_preprocessed)
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    vectors = vectorizer.fit_transform([target_info_preprocessed] + sections_preprocessed)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectors)
    target_similarities = similarity_matrix[0, 1:]
    
    return target_similarities

def find_relevant_sections(target_info, all_sections, threshold=0.5):
    similarities = calculate_similarity(target_info, all_sections)
    relevant_sections = [section for section, similarity in zip(all_sections, similarities) if similarity > threshold]
    return relevant_sections





def sentiment(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    tagged_words = [pos_tag(word) for word in words]
    chunked_sentences = [ne_chunk(tagged_sentence) for tagged_sentence in tagged_words]
    # Define keywords related to the statement
    statement_keywords = ["central", "contributions", "significance", "advancing", "disciplinary", "understandings"]

    # Initialize Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # Search for statements similar to the example
    for sentence in chunked_sentences:
        sentence_text = ' '.join(word for word, _ in sentence.leaves())
        
        # Check if sentence contains the statement keywords
        if all(keyword in sentence_text for keyword in statement_keywords):
            sentiment_score = sia.polarity_scores(sentence_text)["compound"]
            if sentiment_score > 0.5:  # Example threshold for positive sentiment
                print("Potential statement:", sentence_text)

##refactor into individual trycatch
def installnltk():
    #nltk.download('omw-1.4')
    #nltk.download('maxent_ne_chunker')
    #nltk.download('words')
    #nltk.download('punkt')
    #nltk.download('stopwords')
    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('wordnet')
    #nltk.download('vader_lexicon')
    nltk.download('all')

def checknltk():
    try:
        #nltk.data.find('omw-1.4')
        #nltk.data.find('maxent_ne_chunker')
        #nltk.data.find('words')
        #nltk.data.find('punkt')
        #nltk.data.find('stopwords')
        #nltk.data.find('averaged_perceptron_tagger')
        #nltk.data.find('wordnet')
        #nltk.data.find('vader_lexicon')
        nltk.data.find('all')
        print("nltk present")
    except LookupError:
        installnltk()
        print("nltk installed")
#bad
def check_nltk_libraries():
    # Check if all the nltk_data are already downloaded
    nltk_data = ['omw-1.4', 'vader_lexicon', 'maxent_ne_chunker', 'words', 'punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    if all(nltk.find(data) for data in nltk_data):
        print("All NLTK libraries are installed")
    else:
        print("Downloading NLTK libraries")
        # If not, download them all
        nltk.download(nltk_data)
        print("NLTK libraries have been downloaded")
#bad
def search_information(text, guidelines):
    # Tokenize the text and remove stopwords
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
    
    # Search for guidelines in the tokenized words
    found_guidelines = []
    for guideline in guidelines:
        print(guideline)
        guideline_words = word_tokenize(guideline.lower())
        if all(word in words for word in guideline_words):
            found_guidelines.append(guideline)
    
    return found_guidelines
#bad
def printaccuracy(accuracy,guideline_results):
    print(f"Accuracy: {accuracy:.2%}")
    for guideline, (guideline_found, matching_sentences) in guideline_results.items():
        print(f"Guideline: {guideline}")
        print(f"Accuracy: {'Found' if guideline_found else 'Not found'}")
        if matching_sentences:
            print("Matching Sentences:")
            for sentence in matching_sentences:
                print(sentence)
        print("=" * 40)

    
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

# initialize NLTK sentiment analyzer




# create get_sentiment function

def get_sentiment(text):
    
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)

    sentiment = 1 if scores['pos'] > 0 else 0

    return sentiment



def texttiling(text):
    from nltk import tokenize
    ttt=tokenize.TextTilingTokenizer(w=2,k=5,cutoff_policy="loose")
    return ttt.tokenize(text)
    

def siamese(dataset):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Sample dataset

    # Vectorize the dataset
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(dataset)

    # Define the number of clusters
    k = 2

    # Create a k-means model and fit it to the data
    km = KMeans(n_clusters=k)
    km.fit(X)

    # Predict the clusters for each document
    y_pred = km.predict(X)



def loadtext(relativepath):
    with open(os.path.join(PATH , relativepath),encoding="utf8") as file:
        text_data = file.read()
    return text_data


import Levenshtein


def find_start_index_of_closest_substring(text, substring):
    min_distance = float('inf')
    start_index = -1

    for i in range(len(text) - len(substring) + 1):
        candidate_substring = text[i:i + len(substring)]
        distance = Levenshtein.distance(substring, candidate_substring)

        if distance < min_distance:
            min_distance = distance
            start_index = i

    return start_index


import threading


def worker(args):
    tokenized_line, guideline = args
    comparebasic(tokenized_line, guideline)



def extract_info(text):
    return " ".join([token.lemma_ for token in text if token.pos_ in ['NOUN', 'VERB']])


import numpy as np


from gensim.models import Word2Vec

import langid
def filter_english_text(text):
    # Use langid to detect the language
    lang, confidence = langid.classify(text)

    # If the detected language is English with sufficient confidence, keep the text
    if lang == 'en' and confidence > 0.8:
        return text
    else:
        return None
    
#def most_common(lst):
 #   return max(set(lst), key=lst.count)

def most_common(lst, n=3):
    data = Counter(lst)
    most_common_elements = [item[0] for item in data.most_common(n)]
    return most_common_elements

def calculate_similarity(tokens_text, tokens_rule):
    common_tokens = set(tokens_text).intersection(set(tokens_rule))
    total_tokens = len(set(tokens_text).union(set(tokens_rule)))

    similarity = len(common_tokens) / total_tokens if total_tokens != 0 else 0
    return similarity



def driver(text_data,guidelinepath,documentobject=None,debug=False):
    try:
        nlp=load("en_use_md")
        #checknltk()
        def process_batch(batch):
            similarities = []
            
            for text,rule in batch:
                proctext = nlp(text)
                procrule = nlp(rule)
                similarity = proctext.similarity(procrule)
                similarities.append(similarity)
                #print(f"text: {proctext}\nrule: {procrule}\nsimilarity: {similarity}\n\n")
            return similarities
        
            
        def process_batchalt(batch):
            similarities = []
            
            for text,rule in batch:

                
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([text, rule])

                # Calculate cosine similarity between the vectors
                similarity_matrix = cosine_similarity(vectors)
                
                # The similarity score is at position [0, 1] or [1, 0] in the matrix
                similarity = max(similarity_matrix[0,1], similarity_matrix[1,0])

                similarities.append(similarity)
                #print(f"text: {text}\nrule: {rule}\nsimilarity: {similarity}\n\n")
            return similarities
        
        
        def process_batchalttwo(batch):
            similarities = []
            #lemmatizer =WordNetLemmatizer()
            for text, rule in batch:
                
                #tokens_text = [lemmatizer.lemmatize(token.text.lower()) for token in nlp(text)]

                
                #tokens_rule = [lemmatizer.lemmatize(token.text.lower()) for token in nlp(rule)]

                
                similarity = calculate_similarity(text, rule)
                similarities.append(similarity)

                
                print(f"text: {text}\nrule: {rule}\nsimilarity: {similarity}\n\n")

            return similarities
        
        
        guidelinedict=read_file(os.path.join(PATH,guidelinepath))
        guidelines=extract_sentences(guidelinedict)
        testguidelines=[guideline[3] for guideline in guidelines]
        actuallines=[preprocess_text(guideline[3]) for guideline in guidelines]
        #print(actuallines)
        #quantdict=read_file(os.path.join(PATH,"apaGuidelines/quant.txt"))

        
    
        text_data = loadtext(os.path.join(documentobject.storepage,"entiretext.txt"))
        #remove references#
        from refextract import extract_references_from_file
        references = extract_references_from_file(documentobject.pdfpath)
        #print(references[0])
        for i,reference in enumerate(references):
            #removedref,text = find_and_remove_closest_substring(reference['raw_ref'],text)
            #use algs 
            if reference['raw_ref'][0].lower() in text_data.replace('\n',""):
                position=find_start_index_of_closest_substring(text_data,reference['raw_ref'][0])
                text_data=text_data[:position]
                #temptext_data=text_data.replace('\n',"")
                #text_data=temptext_data.split(reference['raw_ref'][0])[0]
                break
            #text_data=find_and_remove_sentence(text_data,reference['raw_ref'][0])
            #print(reference['raw_ref'])
        #print(text_data.split('.')[::-1])
        #print(text_data)
        #print(text_data)
        tilledtext=texttiling(text_data)
        
        tokenized_lines = [preprocess_text(ttext) for ttext in tilledtext]
        
        
        
        entities,variables= analyse(" ".join(tokenized_lines))
        #[print(tokenized_line+"\n") for tokenized_line in tokenized_lines]
        
        
            # a tokenized line and a guideline
        compare_tasks = [(tokenized_line, guideline) for tokenized_line in tokenized_lines for guideline in actuallines]
        #[print(compare_task) for compare_task in compare_tasks]
        #print(len(compare_tasks))
        #print(len(compare_tasks[0]))
        #print(len(compare_tasks[0][0]))
        
        
        
        
        text_batches = [compare_tasks[i:i + 10] for i in range(0, len(compare_tasks), 10)]


        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(process_batchalt, text_batches))

        similarities = [similarity for batch_result in results for similarity in batch_result]

        similarity_matrix = np.array(similarities).reshape(len(tokenized_lines), len(actuallines))
        
        
    #  print(len(actuallines))
    # print(len(tilledtext))
        #print(similarity_matrix.shape)
        
        
        reporttext=[]
        max_similarity_indices = np.argmax(similarity_matrix, axis=0)
        
        #print(len(max_similarity_indices))
        #print(similarity_matrix.shape)
        for i, max_index in enumerate(max_similarity_indices):
            max_similarity_score = similarity_matrix[max_index, i]
            textpage=documentobject.get_page_number_for_text(tilledtext[max_index])
            reporttext.append([testguidelines[i],textpage,max_similarity_score])
            #print(similarity_matrix[max_index, i], actuallines[i],tilledtext[max_index],textpage)
        
        #print(len(reporttext))
        
        
        #print(reporttext)
        reportdict={}
        reportdict["name"] = documentobject.name
        reportdict["storepage"]=documentobject.storepage
        reportdict["entities"]=most_common(entities)
        reportdict["variables"]=most_common(variables)
        reportdict["similarities"]=reporttext


        reportdict["references"]=references

        
        if debug:
            import matplotlib.pyplot as plt
            
            
            section_names = [f"Section {i + 1}" for i in range(len(tokenized_lines))]

            # Plotting the graph
            plt.figure(figsize=(12, 6))
            
            for i, rule in enumerate(actuallines):
                plt.plot(section_names, similarities[i * len(tokenized_lines):(i + 1) * len(tokenized_lines)], label=f'Rule: {rule}')

            
            
            
            # Adding labels and title
            plt.xlabel('Text Sections')
            plt.ylabel('Similarity Score')
            plt.title('Similarity Scores Across Text Sections for Each Rule')

            # Display legend
            #plt.legend()

            # Display the plot
            plt.grid(True)
            plt.show()
    
    #entitieszip=[analyse(tokenized_line) for tokenized_line in tokenized_lines]
    #print(entitieszip)
    
        import generatereport
        success=generatereport.create_pdf_report(reportdict)
        return success
    
    except Exception as e:
        print(e)
        print("Failed at line:", traceback.extract_tb(e.__traceback__)[-1][1])
        return False
   # 
    
    #printaccuracy(calculate_accuracy(actuallines, tokenized_lines))
    
    
    
    #
    #print(text)
    #for guideline in guidelines:
      #  relevent_sections=find_relevant_sections(guideline[3],text)
    #for section in relevent_sections:
     #   print(section)
    #print(find_paragraph_topic(text))
    #sentiment(text)
    #guidelines=extract_sentences(read_file(os.path.join(PATH,file)))
    #for guideline in guidelines:
       #print(guideline[3])
    
    #sentiment(text)
    
    #print(sentenceguidelines)
    #print(search_information(text,guidelines))
    
    
    

    # Perform named entity recognition

    
    #sent_entity_names,word_entity_names=extract_entities(text)
    #print(sent_entity_names)
    #print(word_entity_names)
    #for quest in guidelines:
        #comparebasic(text,quest[3],debug=True)
    
    
    
    #print(qualdict)
    #print(quantdict)


    # r=classifyabstraction(txt,debug=False)
    
    

#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#embeddings = embed([rule,text])
#print(embeddings)

#https://github.com/monk1337/Awesome-Question-Answering





    


if __name__=="__main__":   
    
    text="""Machine learning (ML) is changing virtually every aspect of our lives. Today ML
    algorithms accomplish tasks that until recently only expert humans could perform.
    As it relates to finance, this is the most exciting time to adopt a disruptive technology
    that will transform how everyone invests for generations. This book explains scientifically sound ML tools that have worked for me over the course of two decades,
    and have helped me to manage large pools of funds for some of the most demanding
    institutional investors.
    
    Books about investments largely fall in one of two categories. On one hand we
    find books written by authors who have not practiced what they teach. They contain
    extremely elegant mathematics that describes a world that does not exist. Just because
    a theorem is true in a logical sense does not mean it is true in a physical sense. On the
    other hand we find books written by authors who offer explanations absent of any
    rigorous academic theory. They misuse mathematical tools to describe actual observations. Their models are overfit and fail when implemented. Academic investigation
    and publication are divorced from practical application to financial markets, and
    many applications in the trading/investment world are not grounded in proper science.
    A first motivation for writing this book is to cross the proverbial divide that separates academia and the industry. I have been on both sides of the rift, and I understand
    how difficult it is to cross it and how easy it is to get entrenched on one side. Virtue is
    in the balance. This book will not advocate a theory merely because of its mathematical beauty, and will not propose a solution just because it appears to work. My goal
    is to transmit the kind of knowledge that only comes from experience, formalized in
    a rigorous manner
    
    A second motivation is inspired by the desire that finance serves a purpose. Over
    the years some of my articles, published in academic journals and newspapers, have
    expressed my displeasure with the current role that finance plays in our society.
    Investors are lured to gamble their wealth on wild hunches originated by charlatans
    and encouraged by mass media. One day in the near future, ML will dominate finance,
    science will curtail guessing, and investing will not mean gambling. I would like the
    reader to play a part in that revolution.
    
    A third motivation is that many investors fail to grasp the complexity of ML applications to investments. This seems to be particularly true for discretionary firms moving into the “quantamental” space. I am afraid their high expectations will not be
    met, not because ML failed, but because they used ML incorrectly. Over the coming years, many firms will invest with off-the-shelf ML algorithms, directly imported
    from academia or Silicon Valley, and my forecast is that they will lose money (to
    better ML solutions). Beating the wisdom of the crowds is harder than recognizing
    faces or driving cars. With this book my hope is that you will learn how to solve some
    of the challenges that make finance a particularly difficult playground for ML, like
    backtest overfitting. Financial ML is a subject in its own right, related to but separate
    from standard ML, and this book unravels it for you"""

    fabricatedtext="""Title: Exploring the Impact of Artificial Intelligence on Climate Change Modeling

Abstract:
This research investigates the integration of artificial intelligence (AI) techniques in climate change modeling. The study aims to assess the potential benefits of AI in enhancing the accuracy and efficiency of climate models. The methods include data analysis, machine learning algorithms, and predictive modeling. The findings provide insights into the role of AI in addressing challenges related to climate change research.

Introduction:
The introduction outlines the motivation behind incorporating AI into climate change modeling. It discusses the current limitations of traditional modeling approaches and highlights the need for advanced techniques to handle complex environmental data.

Methodology:
The methodology section describes the data collection process, preprocessing steps, and the application of machine learning algorithms. Various AI models, including neural networks and decision trees, are employed to analyze climate data and make predictions.

Results:
The results present the performance metrics of AI-enhanced climate models compared to traditional models. Graphs and visualizations showcase the accuracy improvements and demonstrate the potential of AI in capturing intricate patterns in climate data.

Discussion:
The discussion interprets the results, addressing the implications of using AI in climate change research. It explores the benefits and challenges associated with the integration of machine learning techniques and suggests avenues for future research.

Conclusion:
The conclusion summarizes the key findings and emphasizes the significance of incorporating AI in climate change modeling. It concludes with recommendations for further research to harness the full potential of artificial intelligence in addressing global environmental challenges.

References:
1. Smith, J. et al. (2021). Advances in Climate Modeling Techniques.
2. Jones, A. R. (2022). Machine Learning Applications in Environmental Science.
3. Green, P. Q. (2019). The Role of Artificial Intelligence in Climate Change Research.
4. Brown, B. (2020). Artificial Intelligence and Climate Change: A Review of the Literature.
5. White, W. (2021). Artificial Intelligence and Climate Change: A Systematic Review."""
    storepdfpath="E:/Lvl4Project/ScientificEvidenceReviewApplication/server/testPDF/2022-17092-001.pdf"


    #driver(text,"apaGuidelines\\qual.txt",documentobject=document,debug=True)

    #rule="Identify main variables and theoretical issues under investigation and the relationships between them."
    #text="in his new book Advances in Financial Machine Learning, noted financial scholar Marcos Lopez de Prado strikes a well-aimed karate chop at the naive and often statis-tically overfit techniques that are so prevalent in the financial world today. He points out that not only are business-as-usual approaches largely impotent in today’s high-tech finance, but in many cases they are actually prone to lose money. But Lopez de ´ Prado does more than just expose the mathematical and statistical sins of the finance world. Instead, he offers a technically sound roadmap for finance professionals to join the wave of machine learning. What is particularly refreshing is the author’s empirical approach—his focus is on real-world data analysis, not on purely theoretical methods that may look pretty on paper but which, in many cases, are largely ineffective in practice. The book is geared to finance professionals who are already familiar with statistical data analysis techniques, but it is well worth the effort for those who want to do real state-of-the-art work in the field.”"





#driver(text,"apaGuidelines/qual.txt",debug=True)





