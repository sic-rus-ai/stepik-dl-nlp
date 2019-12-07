import argparse
import pandas as pd
import numpy as np
import string
import pickle
from nltk.util import ngrams


def generate_csv(input_file='predicted_titles.csv', output_file='submission.csv', voc_file='vocs.pkl'):
    '''
    Generates file in format required for submitting result to Kaggle
    
    Parameters:
        input_file (str) : path to csv file with your predicted titles.
                           Should have two fields: abstract and title
        output_file (str) : path to output submission file
        voc_file (str) : path to voc.pkl file
    '''
    data = pd.read_csv(input_file)
    with open(voc_file, 'rb') as voc_file:
        vocs = pickle.load(voc_file)

    with open(output_file, 'w') as res_file:
        res_file.write('Id,Predict\n')
        
    output_idx = 0
    for row_idx, row in data.iterrows():
        trg = row['title']
        trg = trg.translate(str.maketrans('', '', string.punctuation)).lower().split()
        trg.extend(['_'.join(ngram) for ngram in list(ngrams(trg, 2)) + list(ngrams(trg, 3))])
        
        VOCAB_stoi = vocs[row_idx]
        trg_intersection = set(VOCAB_stoi.keys()).intersection(set(trg))
        trg_vec = np.zeros(len(VOCAB_stoi))    

        for word in trg_intersection:
            trg_vec[VOCAB_stoi[word]] = 1

        with open(output_file, 'a') as res_file:
            for is_word in trg_vec:
                res_file.write('{0},{1}\n'.format(output_idx, int(is_word)))
                output_idx += 1
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        help='Path to input .csv file (abstract, title)',
        type=str,
    )
    parser.add_argument(
        '--output_file',
        help='Path to kaggle submission file',
        type=str,
    )
    parser.add_argument(
        '--voc_file',
        help='Path to voc.pkl file',
        type=str,
    )
    args = parser.parse_args()
    generate_csv(args.input_file, args.output_file, args.voc_file)
