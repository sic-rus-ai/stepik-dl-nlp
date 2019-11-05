import torch


import lxml
from lxml import etree

from ipymarkup         import show_box_markup
from ipymarkup.palette import palette, PALETTE, BLUE, RED, GREEN, PURPLE, BROWN, ORANGE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from collections import Counter

# sentiment
PALETTE.set('negative',RED)
PALETTE.set('B-negative',RED)
PALETTE.set('I-negative',RED)
PALETTE.set('positive',GREEN)
PALETTE.set('B-positive',GREEN)
PALETTE.set('I-positive',GREEN)
PALETTE.set('neutral', ORANGE)
PALETTE.set('B-neutral', ORANGE)
PALETTE.set('I-neutral', ORANGE)
PALETTE.set('other', BROWN)
PALETTE.set('Other', BROWN)


# aspect
PALETTE.set('Comfort',ORANGE)
PALETTE.set('B-Comfort',ORANGE)
PALETTE.set('I-Comfort',ORANGE)
PALETTE.set('Appearance',BLUE)
PALETTE.set('B-Appearance',BLUE)
PALETTE.set('I-Appearance',BLUE)
PALETTE.set('Reliability',GREEN)
PALETTE.set('B-Reliability',GREEN)
PALETTE.set('I-Reliability',GREEN)
PALETTE.set('Safety',PURPLE)
PALETTE.set('B-Safety',PURPLE)
PALETTE.set('I-Safety',PURPLE)
PALETTE.set('Driveability',BLUE)
PALETTE.set('B-Driveability',BLUE)
PALETTE.set('I-Driveability',BLUE)
PALETTE.set('Whole',RED)
PALETTE.set('B-Whole',RED)
PALETTE.set('I-Whole',RED)
PALETTE.set('Costs',RED)
PALETTE.set('B-Costs',RED)
PALETTE.set('I-Costs',RED)


def show_markup(text,spans):         
    show_box_markup(text, spans, palette=PALETTE)
    
    
def generate_markup(tokens,tags):
    mapper = lambda tag: tag[2:] if tag!='Other' else tag
    
    tags  = [mapper(tag) for tag in tags]
    text  = ' '.join(tokens)
    spans = []
        
    start,end,tag = 0,len(tokens[0]),tags[0]
    
    for word, ttag in zip(tokens[1:], tags[1:]): 
        
        if tag == ttag:
            end  += 1+len(word)
            
        else:
            span  = (start, end, tag)
            spans.append(span)
        
            start = 1+end
            end  += 1+len(word)
            tag   = ttag
            
    span  = (start, end, tag)
    spans.append(span)        
            
    return text, spans


# --------- tokenization & sentence detection & BIO-tagging ---------


def fill_gaps(text, source_spans):
    """ Заполняем пробелы в авторской разметке """
    
    chunks = []
    
    text_pos = 0
    for span in source_spans:
        s,e,t = span
        if text_pos<s:
            chunks.append((text_pos,s,'Other'))
        chunks.append((s,e,t))
        text_pos=e
            
    if text_pos<len(text)-1:
        chunks.append((text_pos,len(text),'Other'))
    
    return chunks


def extract_BIO_tagged_tokens(text, source_spans, tokenizer):
    """ Разобьем на bio-токены по безпробельной разметке """
    
    tokens_w_tags = []
    for span in source_spans:
        s,e,tag  = span
        tokens   = tokenizer(text[s:e])
        
        if tag == 'Other':
            tokens_w_tags += [(token,tag) for token in tokens]
        else:
            tokens_w_tags.append((tokens[0],'B-'+tag))
            for token in tokens[1:]:
                tokens_w_tags.append((token,'I-'+tag))
                
    return tokens_w_tags


import re

def regex_sentence_detector(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def sentence_spans(text, sentence_detector):
    """ Разбиваем на предложения и генерируем разметку - для красоты """
    
    sentences = sentence_detector(text)

    spans = []
    
    sent_start= 0
    idx       = 1
    for sent in sentences:    
        sent_end = sent_start + len(sent)
        spans.append((sent_start,sent_end, 's{}'.format(idx)))
        
        sent_start = 1+sent_end
        idx       += 1
        
    return spans


def sentence_splitter(text, spans):
    """ Разбиваем на предложения и перегенерируем разметку  """
    result = []
    
    sentences = regex_sentence_detector(text)
    
    sent_start = 0
    span_idx   = 0
    
    for sent in sentences:
        sent_end = sent_start + len(sent)
        
        sent_spans = []
        for span in spans[span_idx:]:
            s,e,t = span
            if e <= sent_end:
                sent_spans.append((s-sent_start,e-sent_start,t))
                span_idx+=1
            else:
                continue
                
        result.append((text[sent_start:sent_end], sent_spans))
        
        sent_start = 1+sent_end

    return result

# --------- tokenization & sentence detection & BIO-tagging ---------
    
    
# ------- SentiRuEval-2015 dataset xml parsing --------------

def get_sentiment_spans(root):
    return get_tag_spans(root,'sentiment')

def get_aspect_spans(root):
    return get_tag_spans(root,'category')

def get_tag_spans(root,tag):
    attributes = root.attrib        
    start = int(attributes['from'])
    end   = int(attributes['to'])
    tag   = attributes[tag]
    span  = (start,end,tag)
    return span


def parse_xml_sentiment(xml_file): return parseXML(xml_file, get_sentiment_spans)
def parse_xml_aspect(xml_file):    return parseXML(xml_file, get_aspect_spans)


def parseXML(xmlFile, span_func):
    xml  = open(xmlFile).read()
    root = etree.fromstring(xml)
    
    result = []
    for review in root.getchildren(): 
        for elem in review.getchildren():            
            if elem.tag == 'text':
                text  = elem.text
            if elem.tag == 'aspects':
                spans = [span_func(xml_span) for xml_span in elem.getchildren()]
                spans = span_sanity_filter(spans)          
        result.append((text, spans))
    return result


def span_sanity_filter(spans):
    result = [spans[0]]
    
    for span in spans[1:]:
        _,prev_span_end,_   = result[-1]
        curr_span_start,_,_ = span
        
        if prev_span_end < curr_span_start:
            result.append(span)
    return result

# ------- SentiRuEval-2015 dataset xml parsing --------------



# ------------ Index & Reverse ------------------------------

class Converter():
    def __init__(self,vocabulary, tags):
        self.idx_to_word = sorted(vocabulary)
        self.idx_to_tag  = sorted(tags)

        self.word_to_idx = {word:idx for idx,word in enumerate(self.idx_to_word)}
        self.tag_to_idx  = { tag:idx for idx,tag  in enumerate(self.idx_to_tag)}
        
    def words_to_index(self, words):
        return torch.tensor([self.word_to_idx[w] for w in words], dtype=torch.long)
    
    def tags_to_index(self, words):
        return torch.tensor([self.tag_to_idx[w] for w in words], dtype=torch.long)
    
    def indices_to_words(self, indices):
        return [self.idx_to_word[i] for i in indices]
    
    def indices_to_tags(self, indices):
        return [self.idx_to_tag[i] for i in indices]
    

def form_vocabulary_and_tagset(words_w_tags):
    dictionary = set()
    tagset     = set()
    
    for words,tags in words_w_tags: 
        for word, tag in zip(words, tags):
            dictionary.add(word)
            tagset.add(tag)
            
    return dictionary,tagset


def prepare_data(texts_with_spans, tokenize_func):
    result = []
    
    for text, spans in texts_with_spans:
        for sent, sent_spans in sentence_splitter(text, spans):
            if len(sent)>1:
                cover_spans      = fill_gaps(sent, sent_spans)                
                try:
                    tokens_w_biotags = extract_BIO_tagged_tokens(sent, cover_spans, tokenize_func)
                    tokens, biotags  = list(zip(*tokens_w_biotags))
                    result.append((tokens, biotags))
                    
                except Exception as e:
                    continue
                    
    return result

# ------------ Index & Reverse ------------------------------

# ----------------- Metrics ---------------------------------
def predict_tags(model, converter, recipe):
    encoded_recipe = converter.words_to_index(recipe)        # слово -> его номер в словаре
    encoded_tags   = model.predict_tags(encoded_recipe)      # предсказанные тэги (номера)
    decoded_tags   = converter.indices_to_tags(encoded_tags) # номер тэга -> тэг
    return decoded_tags


def tag_statistics(model, converter, data):

    def tag_counter(predicted, ground):
        correct_tags = Counter()
        ground_tags  = Counter(ground)
        
        for tag_p, tag_g in zip(predicted, ground):
            if tag_p==tag_g:
                correct_tags[tag_g]+=1            
        return correct_tags, ground_tags
    
    
    total_correct, total_tags = Counter(), Counter()
    
    for recipe, tags in data:
        tags_pred              = predict_tags(model, converter, recipe)
        tags_correct, tags_num = tag_counter(tags_pred, tags)

        total_correct.update(tags_correct)
        total_tags.update(tags_num)

    return total_correct, total_tags


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
   
    cm     = confusion_matrix(y_true, y_pred, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im      = ax.imshow(cm, interpolation='nearest', cmap=cmap)
   
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, 
           yticklabels=classes,
           title=title,
           ylabel='Истинный тэг',
           xlabel='Предсказанный тэг')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax
# ----------------- Metrics ---------------------------------