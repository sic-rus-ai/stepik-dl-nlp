import argparse
import pandas as pd
import numpy as np
import string
from nltk.util import ngrams


def generate_csv(input_file='test.csv', output_file='submission.csv'):
    '''
    Generates file in format required for submitting result to Kaggle
    
    Parameters:
        input_file (str) : path to csv file with your predicted titles.
                           Should have two fields: abstract and title
        output_file (str) : path to output submission file
    '''
    data = pd.read_csv(input_file)

    with open(output_file, 'w') as res_file:
        res_file.write('Id,Predict\n')
        
    output_idx = 0
    for row_idx, row in data.iterrows():
        src = row['abstract']
        src = src.translate(str.maketrans('', '', string.punctuation)).lower().split()
        src.extend(['_'.join(ngram) for ngram in list(ngrams(src, 2)) + list(ngrams(src, 3))])

        trg = row['title']
        trg = trg.translate(str.maketrans('', '', string.punctuation)).lower().split()
        trg.extend(['_'.join(ngram) for ngram in list(ngrams(trg, 2)) + list(ngrams(trg, 3))])

        VOCAB = set(src)
        VOCAB_stoi = {word : i for i, word in enumerate(VOCAB)}

        trg_intersection = set(VOCAB).intersection(set(trg))
        trg_vec = np.zeros(len(VOCAB))    

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
    args = parser.parse_args()
    generate_csv(args.input_file, args.output_file)

