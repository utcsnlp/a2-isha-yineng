# models.py
import math

from optimizers import *
from nerdata import *
from utils import *
import random
import time

from collections import Counter
from typing import List

import numpy as np
from scipy.special import logsumexp


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs
        self.scorer = ProbabilisticSequenceScorer(tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs)

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        # raise Exception("IMPLEMENT ME")
        V = [{}]
        for tag in range(len(self.tag_indexer)):
            V[0][tag] = {
                "prob": self.scorer.score_init(sentence_tokens, tag) + self.scorer.score_emission(sentence_tokens, tag, 0),
                "prev": None}
        for t in range(1, len(sentence_tokens)):
            V.append({})
            for tag in range(len(self.tag_indexer)):
                if(tag == 4):
                    max_tr_prob = V[t - 1][3]["prob"] + self.scorer.score_transition(sentence_tokens, 3, tag)
                    prev_st_selected = 3
                    tr_prob = V[t - 1][4]["prob"] + self.scorer.score_transition(sentence_tokens, 4, tag)
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = 4

                elif (tag == 6):
                    max_tr_prob = V[t - 1][1]["prob"] + self.scorer.score_transition(sentence_tokens, 1, tag)
                    prev_st_selected = 1
                    tr_prob = V[t - 1][6]["prob"] + self.scorer.score_transition(sentence_tokens, 6, tag)
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = 6

                elif (tag == 7):
                    max_tr_prob = V[t - 1][2]["prob"] + self.scorer.score_transition(sentence_tokens, 2, tag)
                    prev_st_selected = 2
                    tr_prob = V[t - 1][7]["prob"] + self.scorer.score_transition(sentence_tokens, 7, tag)
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = 7
                elif (tag == 8):
                    max_tr_prob = V[t - 1][5]["prob"] + self.scorer.score_transition(sentence_tokens, 5, tag)
                    prev_st_selected = 5
                    tr_prob = V[t - 1][8]["prob"] + self.scorer.score_transition(sentence_tokens, 8, tag)
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = 8

                else:
                    max_tr_prob = V[t - 1][0]["prob"] + self.scorer.score_transition(sentence_tokens, 0, tag)
                    prev_st_selected = 0
                    for prev_st in range(1, len(self.tag_indexer)):
                        tr_prob = V[t - 1][prev_st]["prob"] + self.scorer.score_transition(sentence_tokens, prev_st, tag)
                        if tr_prob > max_tr_prob:
                            max_tr_prob = tr_prob
                            prev_st_selected = prev_st


                max_prob = max_tr_prob + self.scorer.score_emission(sentence_tokens, tag, t)
                V[t][tag] = {"prob": max_prob, "prev": prev_st_selected}

        opt = []
        max_prob = -float('inf')
        best_st = None

        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(self.tag_indexer.get_object(best_st))
        previous = best_st

        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, self.tag_indexer.get_object(V[t + 1][previous]["prev"]))
            previous = V[t + 1][previous]["prev"]

        # print("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(opt))



def train_hmm_model(sentences: List[LabeledSentence], silent: bool=False) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    if not silent:
        print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    if not silent:
        print("Tag indexer: %s" % tag_indexer)
        print("Initial state log probabilities: %s" % init_counts)
        print("Transition log probabilities: %s" % transition_counts)
        print("Emission log probs too big to print...")
        print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
        print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
        print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


##################
# CRF code follows

class FeatureBasedSequenceScorer(object):
    """
    Feature-based sequence scoring model. Note that this scorer is instantiated *for every example*: it contains
    the feature cache used for that example.
    """
    def __init__(self, tag_indexer, feature_weights, feat_cache):
        self.tag_indexer = tag_indexer
        self.feature_weights = feature_weights
        self.feat_cache = feat_cache

    def score_init(self, sentence, tag_idx):
        if isI(self.tag_indexer.get_object(tag_idx)):
            return -1000
        else:
            return 0

    def score_transition(self, sentence_tokens, prev_tag_idx, curr_tag_idx):
        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
        curr_tag = self.tag_indexer.get_object(curr_tag_idx)

        # rule, I-XXX must follows a B-XXX or I-XXX
        if isI(curr_tag):
            if (isI(prev_tag) or isB(prev_tag)) and get_tag_label(prev_tag) == get_tag_label(curr_tag):
                return 0
            else:
                return -1000  # impossible, so give a huge penalty
        return 0

    def score_emission(self, sentence_tokens, tag_idx, word_posn):
        feats = self.feat_cache[word_posn][tag_idx]
        score = 0
        for f in feats:
            score += self.feature_weights[f][tag_idx]

        return score


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, feature_cache):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.feature_cache = feature_cache
        self.scorer = None


    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        num_tokens = len(sentence_tokens)
        num_tags = len(self.tag_indexer)
        viterbi = np.zeros((num_tags, num_tokens))  # a matrix for dynamic programming
        backpointer = np.zeros((num_tags, num_tokens))

        # if feature_cache is None, need to extract feature first
        if self.feature_cache is None:
            feature_cache = [[[] for k in range(num_tags)] for j in range(num_tokens)]
            for word_idx in range(num_tokens):
                for tag_idx in range(num_tags):
                    feature_cache[word_idx][tag_idx] = extract_emission_features(sentence_tokens, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer=False)
            self.update_scorer(feature_cache)

        # initialization step
        for tag_idx in range(num_tags):
            viterbi[tag_idx][0] = self.scorer.score_init(sentence_tokens, tag_idx) + self.scorer.score_emission(sentence_tokens, tag_idx, 0)
            backpointer[tag_idx][0] = -1  # -1 for invalid idx

        # recursion step
        for cur_token in range(1, num_tokens):
            for cur_tag in range(num_tags):
                max_score = float('-inf')
                arg_max = -1

                emission_score = self.scorer.score_emission(sentence_tokens, cur_tag, cur_token)  # move out of loop to avoid recompute
                for prev_tag in range(num_tags):
                    transition_score = self.scorer.score_transition(sentence_tokens, prev_tag, cur_tag)
                    score = viterbi[cur_tag][cur_token-1] + emission_score + transition_score
                    if score > max_score:
                        max_score = score
                        arg_max = prev_tag

                viterbi[cur_tag][cur_token] = max_score
                backpointer[cur_tag][cur_token] = arg_max

        # find the tags by tracing backpointer
        bio_tags = [""] * num_tokens

        last_tag = np.argmax(viterbi, axis=0)[-1]
        bio_tags[-1] = self.tag_indexer.get_object(last_tag)

        last_tag = int(backpointer[last_tag][-1])

        for cur_token in range(num_tokens-2, -1, -1):
            bio_tags[cur_token] = self.tag_indexer.get_object(last_tag)
            last_tag = int(backpointer[last_tag][cur_token])

        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(bio_tags))


    def decode_beam(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        raise Exception("IMPLEMENT ME")

    def get_forward_backward(self, sentence_tokens: List[Token]):
        """
        This method will compute forward and backward array,
        Should be call for every sentence.
        Caller could use Forward-Backward algorithm to compute P(s_t = n | x).
        Note: everything is computed in log space to avoid underflow
        """
        num_tokens = len(sentence_tokens)
        num_tags = len(self.tag_indexer)
        forward = np.zeros((num_tags, num_tokens))
        backward = np.zeros((num_tags, num_tokens))

        #--- Forward Part ---
        # initialization step
        for tag_idx in range(num_tags):
            forward[tag_idx][0] = self.scorer.score_init(sentence_tokens, tag_idx) + self.scorer.score_emission(sentence_tokens, tag_idx, 0)

        # recursion step
        for cur_token in range(1, num_tokens):
            for cur_tag in range(num_tags):
                raw_value = []
                emission_score = self.scorer.score_emission(sentence_tokens, cur_tag, cur_token)  # move out of loop to avoid recompute
                for prev_tag in range(num_tags):
                    raw_value.append(forward[prev_tag][cur_token-1] + emission_score + self.scorer.score_transition(sentence_tokens, prev_tag, cur_tag))
                forward[cur_tag][cur_token] = logsumexp(raw_value)

        #--- Backward Part ---
        # initialization step
        for tag_idx in range(num_tags):
            backward[tag_idx][-1] = 1

        # recursion step
        for cur_token in range(num_tokens - 2, -1, -1):
            for cur_tag in range(num_tags):
                raw_value = []
                for next_tag in range(num_tags):
                    raw_value.append(backward[next_tag][cur_token+1] +
                                     self.scorer.score_emission(sentence_tokens, next_tag, cur_token+1) + self.scorer.score_transition(sentence_tokens, cur_tag, next_tag))
                backward[cur_tag][cur_token] = logsumexp(raw_value)

        return forward, backward  # back to real space

    def compute_gradient(self, labeled_sentence: LabeledSentence, sentence_idx: int):
        """
        sentence with gold (sentence.tokens[0].chunk) and predicted labels (sentence.chunks)
        """
        num_tokens = len(labeled_sentence.tokens)
        bio_tags = bio_tags_from_chunks(labeled_sentence.chunks, num_tokens)  # the predicted labels
        num_tags = len(self.tag_indexer)
        features = self.feature_cache[sentence_idx]

        # use to compute marginal probabilities P(s_t = n | x).
        forward, backward = self.get_forward_backward(labeled_sentence.tokens)

        feature_sum = Counter()
        expectation = Counter()

        for token_idx in range(num_tokens):
            # emission feature sum
            tag_idx = self.tag_indexer.index_of(bio_tags[token_idx])
            feature_sum.update(features[token_idx][tag_idx])

            # emission feature expectation (apply forward-backward algorithm here)
            # To compute P(y_i = s | x)
            # the denominator is the same for each i
            # note the matrix's value is in log space, so the computing below should also in log space
            log_product = forward.transpose()[token_idx] + backward.transpose()[token_idx]
            denominator = logsumexp(log_product)
            for s in range(num_tags):
                P = np.exp(log_product[s] - denominator)

                gold_tag_idx = self.tag_indexer.index_of(labeled_sentence.tokens[token_idx].chunk)
                f = Counter()

                for k in features[token_idx][gold_tag_idx]:
                    f[k] = P

                expectation.update(f)
        feature_sum.subtract(expectation)
        return feature_sum

    def update_scorer(self, feature_cache):
        self.scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_weights, feature_cache)

def train_crf_model(sentences: List[LabeledSentence], silent: bool=False) -> CrfNerModel:
    """
    Trains a CRF NER model on the given corpus of sentences.
    :param sentences: The training data
    :param silent: True to suppress output, false to print certain debugging outputs
    :return: The CrfNerModel, which is primarily a wrapper around the tag + feature indexers as well as weights
    """
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    if not silent:
        print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0 and not silent:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    if not silent:
        print("Training")

    # raise Exception("IMPLEMENT THE REST OF ME")
    # initialize CRF
    feature_weights = np.zeros((len(feature_indexer), len(tag_indexer)))
    crf = CrfNerModel(tag_indexer, feature_indexer, feature_weights, feature_cache)
    optimizers = UnregularizedAdagradTrainer(feature_weights)

    num_epoch = 1

    last_time = time.time()
    for _ in range(num_epoch):
        for sentence_idx in range(len(sentences)):
            if sentence_idx % 100 == 0 and not silent:
                cur_time = time.time()
                estimate_total = (cur_time - last_time) * ((len(sentences) - sentence_idx) / 100)
                estimate_sec = round(estimate_total) % 60
                estimate_min = math.floor(estimate_total / 60)
                print("Train %i/%i" % (sentence_idx, len(sentences)))
                print(f"Estimate Time left: {estimate_min} min {estimate_sec} sec")
                last_time = cur_time
            crf.update_scorer(feature_cache[sentence_idx])
            labeled_sentence = crf.decode(sentences[sentence_idx].tokens)
            gradient = crf.compute_gradient(labeled_sentence, sentence_idx)
            optimizers.apply_gradient_update(gradient, 1)

    # clear cache
    crf.feature_cache = None
    return crf


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    return np.asarray(feats, dtype=int)
