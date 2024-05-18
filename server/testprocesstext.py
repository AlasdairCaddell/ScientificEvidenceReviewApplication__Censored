import nltk
from nltk.corpus import wordnet as wn
import os
from fitztest import *
nltk.download('punkt')

PATH=os.path.dirname(os.path.abspath(__file__))

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}

    return tag_dict.get(tag, wn.NOUN)

def find_closest_words(word, topn=5):
    pos = get_wordnet_pos(word)
    synsets = wn.synsets(word, pos)


    max_similarity = {}
    for synset in synsets:
        for lemma in synset.lemmas():
            if lemma.name() == word:
                continue
            similarity = synset.path_similarity(wn.synset(lemma.name()))
            if similarity is not None:
                max_similarity[lemma.name()] = max(max_similarity.get(lemma.name(), 0), similarity)


    sorted_words = sorted(max_similarity.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:topn]]


def synonym_antonym_extractor(phrase):
     from nltk.corpus import wordnet
     synonyms = []
     antonyms = []

     for syn in wordnet.synsets(phrase):
          for l in syn.lemmas():
               synonyms.append(l.name())
               if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())

     print(set(synonyms))
     print(set(antonyms))




def loadtext():
    with open(os.path.join(PATH , "sectionimg\entiretext.txt"),encoding="utf8") as file:
        text_data = file.read()
    return text_data

def loaddummypdffitz():
    pdf_path = os.path.join(PATH , "testPDF\Generatedpdfpopulated.pdf")
    text_data = extract_text_from_pdf(pdf_path)
    return text_data


def preprocess(text):
    # Tokenize into sentences and preprocess
    #preprocessed_sentences = [sentence.lower() for sentence in text]
    tokenized_lines = nltk.word_tokenize(text)
    #print(tokenized_lines)
    return tokenized_lines

def are_synonyms(term1, term2):

    synsets1 = wn.synsets(term1)
    synsets2 = wn.synsets(term2)

    # Check if there's any common synset
    common_synsets = set(synsets1) & set(synsets2)
    return bool(common_synsets)


def texttest(text_data, section_keywords):
    #with open('your_large_text.txt', 'r', encoding='utf-8') as file:
    #    text_data = file.read()
    #        
        
    sentences = (nltk.sent_tokenize(text_data))

    #dict sttructure [guideline_name:actualname] = [abstract:abstract, introduction:motivation, methodology:False, results:False, conclusion:False]
    
    section_names = []

    
   #print(sentences)
    
    
    for keyword in section_keywords:
        are_synonyms(keyword, keyword)

    
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        for keyword in section_keywords:
            
            if keyword in words and len(words.split(" ")) < 3:
                section_names.append(keyword)
                break
            
    
    
    
            
    print(section_names)        
            
    for i in range(len(section_names)):
        if section_names[i] == 'abstract':
            # Perform contextual analysis to confirm the section boundary
            # Adjust or remove if necessary
            print('Abstract section found at index', i)
        

        
    section_boundaries = {}
    current_section = ''
    for i, sentence in enumerate(sentences):
        
        for section_name in section_keywords:
            if section_name in sentence.lower():
                if section_name != current_section:
                    current_section = section_name
                    section_boundaries[current_section] = [i]
                else:
                    section_boundaries[current_section].append(i)
    
    
    section_boundaries = {}
    current_section = ''

    section_texts = {}

    for section, indices in section_boundaries.items():
        section_texts[section] = ' '.join(sentences[min(indices): max(indices) + 1])


    for section, text in section_texts.items():
        print(f'{section.capitalize()}: {text}\n')
 
 
 #===================================================================================
# import libraries
import pandas as pd

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer


# download nltk corpus (first time only)
import nltk

#nltk.download('all')




# Load the amazon review dataset

#df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

# create preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

# initialize NLTK sentiment analyzer

analyzer = SentimentIntensityAnalyzer()


# create get_sentiment function

def get_sentiment(text):

    scores = analyzer.polarity_scores(text)

    sentiment = 1 if scores['pos'] > 0 else 0

    return sentiment






def texttiling(text):
    from nltk import tokenize
    ttt=tokenize.TextTilingTokenizer(w=3,k=5,cutoff_policy="loose")
    return ttt.tokenize(text)


    














    

def test2():
    text_data = loadtext()
    #print(text_data)
    tilledtext=texttiling(text_data)
    tokenized_lines = [preprocess_text(ttext) for ttext in tilledtext]
    [print(tokenized_line+"\n") for tokenized_line in tokenized_lines]

    
 
    
    #section_keywords = ['Abstract', 'Introduction', 'Methodology', 'Results', 'Conclusion']
    #texttest(tokenized_lines,section_keywords)    

         
if __name__ == '__main__':
    

    text = """Abstract: 
    This research explores the impact of artificial intelligence on modern business practices. The study delves into the methodologies employed, the results obtained, and concludes with insights into the future implications of AI integration.

    Introduction: 
    In the ever-evolving landscape of technology, artificial intelligence has emerged as a transformative force. This section introduces the research focus, outlines the objectives, and sets the stage for a comprehensive examination of AI's role in business.

    Methodology: 
    To conduct a thorough analysis, a multi-faceted methodology was employed. Data collection involved surveys, interviews, and real-world case studies. The steps taken in the research process are detailed in this section.

    Results: 
    Presenting the outcomes of the study, this section showcases the tangible impacts of AI on businesses. From improved efficiency to enhanced decision-making, the results highlight the diverse benefits witnessed across various sectors.

    Conclusion: 
    Drawing insights from the research findings, the conclusion section summarizes the key takeaways. It also discusses potential future developments and areas for further exploration in the dynamic realm of artificial intelligence.

    Acknowledgments: 
    The researchers extend gratitude to those who contributed to the success of this study, including participants, advisors, and collaborators.

    References: 
    1. Smith, J. (2020). The Rise of AI in Business. Journal of Technology Impact, 15(2), 45-58. 
    2. Brown, A., et al. (2019). Transformative Technologies: AI and Beyond. Oxford University Press.""" 

    
    test2()