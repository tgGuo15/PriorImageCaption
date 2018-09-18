import json
import numpy as np
import math
from collections import defaultdict


def transfer_result_to_res(data):
    res = {}
    for i in range(len(data)):
        res[data[i]['image_id']] = [data[i]['caption']]
    return res


def transfer_json_to_cider_gts(json_file):
    print '... changing standard format for cider calculation'
    with open(json_file) as f:
         data = json.load(f)
    image_index = data['image_ids']
    index_caption = data['captions']
    gts_caption = {}
    for i in range(len(image_index)):
        gts_caption[image_index[i]] = index_caption[i]
    print '... finishing changing standard format'
    return gts_caption


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1,n+1):
        for i in xrange(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)


class CiderScorer(object):
    def __init__(self, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ref_to_imageId = {}
        self.build_cook_refs(refs)
        self.document_frequency = defaultdict(float)
        self.compute_doc_freq()

    def compute_doc_freq(self):
        """
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        """
        print 'done for stats'
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.iteritems()]):
                self.document_frequency[ngram] += 1
                # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def Counts2vec(self,cnts):
        """
        Function maps counts of ngram to vector of tfidf weights.
        The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
        The n-th entry of array denotes length of n-grams.
        :param cnts:
        :return: vec (array of dict), norm (array of float), length (int)
        """
        vec = [defaultdict(float) for _ in range(self.n)]
        length = 0
        norm = [0.0 for _ in range(self.n)]
        for (ngram, term_freq) in cnts.iteritems():
            # give word count 1 if it doesn't appear in reference corpus
            df = np.log(max(1.0, self.document_frequency[ngram]))
            # ngram index
            n = len(ngram) - 1
            # tf (term_freq) * idf (precomputed idf) for n-grams
            vec[n][ngram] = float(term_freq) * (self.ref_len - df)
            # compute norm for the vector.  the norm will be used for computing similarity
            norm[n] += pow(vec[n][ngram], 2)

            if n == 1:
                length += term_freq
        norm = [np.sqrt(n) for n in norm]
        return vec, norm, length

    def Sim(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
        """
        Compute the cosine similarity of two vectors.
        :param vec_hyp: array of dictionary for vector corresponding to hypothesis
        :param vec_ref: array of dictionary for vector corresponding to reference
        :param norm_hyp: array of float for vector corresponding to hypothesis
        :param norm_ref: array of float for vector corresponding to reference
        :param length_hyp: int containing length of hypothesis
        :param length_ref: int containing length of reference
        :return: array of score for each n-grams cosine similarity
        """
        delta = float(length_hyp - length_ref)
        # measure consine similarity
        val = np.array([0.0 for _ in range(self.n)])
        for n in range(self.n):
            # ngram
            for (ngram, count) in vec_hyp[n].iteritems():
                # vrama91 : added clipping
                val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

            if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                val[n] /= (norm_hyp[n] * norm_ref[n])

            assert (not math.isnan(val[n]))
            # vrama91: added a length based gaussian penalty
            val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
        return val

    # compute log reference length

    def build_cook_refs(self, refs):
        count = 0
        if refs is not None:
            for item in refs:
                 self.ref_to_imageId[item] = count
                 self.crefs.append(cook_refs(refs[item], n= self.n))
                 count = count + 1

    def cook_append_test(self, test=None):
        self.ctest = []
        self.test_to_imageId = {}
        Counttest = 0
        if test is not None:
            for item in test:
                self.test_to_imageId[Counttest] = item
                self.ctest.append(cook_test(test[item][0], n=self.n))
                Counttest = Counttest + 1
        else:
            self.ctest.append(None)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.iteritems():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram) - 1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram, count) in vec_hyp[n].iteritems():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])

                assert (not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        # for test, refs in zip(self.ctest, self.crefs):
        for id in range(len(self.ctest)):
            test = self.ctest[id]
            refs = self.crefs[self.ref_to_imageId[self.test_to_imageId[id]]]
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        #if first_time == 1:

        # assert to check document frequency
        #assert (len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)