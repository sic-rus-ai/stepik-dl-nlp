import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .base import PAD_TOKEN


def get_answer_token_positions(paragraph_text, tokenized_paragraph, answer_text):
    if len(answer_text) == 0:
        return None

    answer_start_offset = paragraph_text.find(answer_text)
    if answer_start_offset < 0:
        return None

    answer_end_offset = answer_start_offset + len(answer_text)

    start_token_i = -1
    end_token_i = -1
    for token_i, (token_str, (token_start_offset, token_end_offset)) in enumerate(tokenized_paragraph):
        if start_token_i == -1 and answer_start_offset <= token_start_offset:
            start_token_i = token_i
        if end_token_i == -1 and answer_end_offset <= token_end_offset:
            end_token_i = token_i

    if start_token_i < 0 or end_token_i < 0:
        return None

    return start_token_i, end_token_i


class SQuADDataset(Dataset):
    def __init__(self, corpus_tokenizer, word2id, paragraphs, questions, answers=None):
        self.word2id = word2id

        assert len(paragraphs) == len(questions)
        self.paragraphs = paragraphs
        self.paragraph_tokens = corpus_tokenizer(paragraphs)

        self.questions = questions
        self.question_tokens = corpus_tokenizer(questions)

        if answers is not None:
            assert len(answers) == len(questions)

            bad_examples_n = 0

            valid_paragraphs = []
            valid_questions = []
            valid_answer_positions = []
            for i, answer_text in enumerate(answers):
                answer_positions = get_answer_token_positions(self.paragraphs[i], self.paragraph_tokens[i], answer_text)
                if answer_positions is None:
                    bad_examples_n += 1
                    continue
                valid_paragraphs.append(self.paragraph_tokens[i])
                valid_questions.append(self.question_tokens[i])
                valid_answer_positions.append(answer_positions)

            self.paragraph_tokens = valid_paragraphs
            self.question_tokens = valid_questions
            self.answer_positions = valid_answer_positions

            if bad_examples_n > 0:
                print('Не получилось сопоставить ответы с текстом в {} случаях'.format(bad_examples_n))
        else:
            self.answer_positions = None

        self.max_paragraph_len = max(len(tokens) for tokens in self.paragraph_tokens)
        self.max_question_len = max(len(tokens) for tokens in self.question_tokens)

    def __len__(self):
        return len(self.paragraph_tokens)

    def __getitem__(self, item):
        paragraph_tokens = self.paragraph_tokens[item]
        paragraph_len = len(paragraph_tokens)
        if paragraph_len < self.max_paragraph_len:
            paragraph_tokens += [(PAD_TOKEN, (-1, -1))] * (self.max_paragraph_len - paragraph_len)

        question_tokens = self.question_tokens[item]
        question_len = len(question_tokens)
        if question_len < self.max_question_len:
            question_tokens += [(PAD_TOKEN, (-1, -1))] * (self.max_question_len - question_len)

        paragraph_word_ids = np.array([self.word2id[token] for token, _ in paragraph_tokens if token in self.word2id])
        question_word_ids = np.array([self.word2id[token] for token, _ in question_tokens if token in self.word2id])

        if self.answer_positions is not None:
            return ((paragraph_word_ids, np.array(paragraph_len),
                     question_word_ids, np.array(question_len)),
                    np.array(self.answer_positions[item], dtype='long'))
        else:
            return ((paragraph_word_ids, np.array(paragraph_len),
                     question_word_ids, np.array(question_len)),
                    0)


def get_top_start_end_pairs(scores, topk=1, max_answer_length=15):
    """scores - Len x 2"""
    max_len = scores.shape[0]
    start_scores = scores[:, 0]
    end_scores = scores[:, 1]
    joint_scores = start_scores[:, None] * end_scores[None, :]  # Len x Len
    joint_scores.triu_().tril_(max_answer_length)

    joint_scores_flat = joint_scores.view(-1)
    top_scores_flat, top_idx_flat = joint_scores_flat.topk(topk)

    start_positions = top_idx_flat // max_len
    end_positions = top_idx_flat - (start_positions * max_len)
    return list(zip(start_positions.cpu().numpy(),
                    end_positions.cpu().numpy(),
                    top_scores_flat.cpu().numpy()))


class AnswerFinder:
    def __init__(self, corpus_tokenizer, word2id, model, topk=2, max_answer_length=15, device='cuda'):
        self.corpus_tokenizer = corpus_tokenizer
        self.word2id = word2id
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.topk = topk
        self.max_answer_length = max_answer_length

    def __call__(self, paragraph, question):
        paragraph_tokens = self.corpus_tokenizer([paragraph])[0]
        question_tokens = self.corpus_tokenizer([question])[0]

        paragraph_word_ids = np.array([self.word2id[token] for token, _ in paragraph_tokens if token in self.word2id])
        question_word_ids = np.array([self.word2id[token] for token, _ in question_tokens if token in self.word2id])

        paragraph_batch = torch.from_numpy(paragraph_word_ids).unsqueeze(0).to(self.device)
        question_batch = torch.from_numpy(question_word_ids).unsqueeze(0).to(self.device)

        paragraph_len_batch = torch.tensor([len(paragraph_word_ids)], device=self.device)
        question_len_batch = torch.tensor([len(question_word_ids)], device=self.device)

        with torch.no_grad():
            answer_logits = self.model((paragraph_batch, paragraph_len_batch,
                                        question_batch, question_len_batch))[0]

        answer_probs = F.softmax(answer_logits, dim=0)

        answer_positions = get_top_start_end_pairs(answer_probs,
                                                   topk=self.topk,
                                                   max_answer_length=self.max_answer_length)

        result = []
        for start, end, score in answer_positions:
            start_offset = paragraph_tokens[start][1][0]
            end_offset = paragraph_tokens[end][1][1]
            result.append((paragraph[start_offset:end_offset], start, end, score))

        return result, answer_probs.cpu().numpy()
