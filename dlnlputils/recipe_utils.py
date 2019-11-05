import numpy as np
import matplotlib.pyplot as plt

from ipymarkup         import show_box_markup
from ipymarkup.palette import palette, PALETTE, BLUE, RED, GREEN, PURPLE, BROWN, ORANGE

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import torch
from collections import Counter

PALETTE.set('NAME', BLUE)
PALETTE.set('UNIT', RED)
PALETTE.set('QTY', GREEN)
PALETTE.set('RANGE_END', GREEN)
PALETTE.set('INDEX', PURPLE)
PALETTE.set('COMMENT', ORANGE)
PALETTE.set('OTHER', BROWN)


def show_markup(recipe,tags):
    mapper = lambda tag: tag[2:] if tag!='OTHER' else tag
    
    tags  = [mapper(tag) for tag in tags]
    text  = ' '.join(recipe)
    spans = []
        
    start,end,tag = 0,len(recipe[0]),tags[0]
    
    for word, ttag in zip(recipe[1:], tags[1:]): 
        
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
            
    show_box_markup(text, spans, palette=PALETTE)
    
    
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


def recipe_statistics(model, converter,data, limit=None):
        
    correct_recipes = Counter()
    total_recipes   = len(data)

    for recipe, tags in data:
        tags_pred = predict_tags(model, converter,recipe)
        
        tags_pred = np.array(tags_pred)
        tags      = np.array(tags)
        noneq_num = np.sum(tags_pred != tags)
        
        if limit and noneq_num>limit:
            title = 'рецепты, размеченные с >{} ошибками'.format(limit)
        elif noneq_num==0:
            title = 'рецепты, размеченные без ошибок'
        else:
            title = 'рецепты, размеченные с {} ошибками:'.format(noneq_num)
            
        
        correct_recipes[title]+=1
    
    return correct_recipes, total_recipes


def plot_recipe_statistics(correct_recipes, total_recipes=None):

    plt.rcdefaults()
    fig, ax = plt.subplots()

    descr,performance = zip(*correct_recipes.most_common())
    y_pos             = np.arange(len(descr))
    
    if total_recipes is not None:
        performance = 100 * (np.array(performance) / float(total_recipes))
        
        ax.set_title('% размеченных рецептов')
        ax.set_xlabel('% рецептов')
        
    else:
        ax.set_title('количество размеченных рецептов')
        ax.set_xlabel('количество рецептов')

    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(descr)
    ax.invert_yaxis()

    plt.show()


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


def form_vocabulary_and_tagset(recipes):
    dictionary = set()
    labels     = set()
    for line in recipes:
        if len(line)>0:
            word, label = line.split('\t')
            dictionary.add(word)
            labels.add(label)
            
    return dictionary,labels


def prepare_data(lines):
    recipes_w_tags = []

    recipe, tags = [],[]
    for line in lines:
        if len(line)>0:
            word, label = line.split('\t')
            recipe.append(word)
            tags.append(label)
        else:
            if len(recipe)>0:
                recipes_w_tags.append((recipe,tags))
            recipe, tags = [],[]

    return recipes_w_tags


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