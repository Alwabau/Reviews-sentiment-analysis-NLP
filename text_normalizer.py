# Importing the necessary libraries
import re
import nltk
import spacy
import string
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
from unidecode import unidecode
nltk.download('stopwords')


tokenizer = ToktokTokenizer()
stopword_list = stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):
    text = text.replace("<br /><br />","")
    return text


def stem_text(text):
    text = tokenizer.tokenize(text)
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    return text


def lemmatize_text(text):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    text = nlp(text)
    text = " ".join([token.lemma_ for token in text])
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    for k, v in contraction_mapping.items():
        text = text.replace(k, v)
    return text


def remove_accented_chars(text):
    text = unidecode(text)
    return text


def remove_special_chars(text, remove_digits=False):
    pattern_punct = r'[' + string.punctuation + ']'
    text = re.sub(pattern_punct, '', text)
    if remove_digits == True:
        pattern_digit = r'[' + string.digits + ']'
        text = re.sub(pattern_digit, '', text) 
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    if is_lower_case == False:
        text = text.lower()   
    text = tokenizer.tokenize(text)
    text = list(filter(lambda x: x not in stopwords, text))
    text = " ".join(text)
    return text


def remove_extra_new_lines(text):
    text = text.replace("\n"," ")
    return text


def remove_extra_whitespace(text):
    text = re.sub(' +', ' ', text)
    return text
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
