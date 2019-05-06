from lenskit.algorithms import Recommender
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import operator
class ItemBasedRecommender(Recommender):

    def __init__(self, beta = 1, alpha = 0, q = 1, test_set_candidates=None):
        self.beta = beta
        self.alpha = alpha
        self.q = q
        self.sim_cache = defaultdict(lambda: defaultdict(float))
        self.ratings = None
        self.items = None
        self.profile_lengths = defaultdict(int)
        self.test_set_candidates = test_set_candidates
        self.processed_users = 0

    def _get_overlap(self, j1, j2):
        '''
        Helper function to get overlap items between two items
        '''
        users_u1 = set(self.ratings.ix[j1].index)
        users_u2 = set(self.ratings.ix[j2].index)
        return list(users_u1.intersection(users_u2))

    def _asym_cosine(self, j1, j2):
        '''
        Helper function: asymmetric cosine similatiry
        '''
        dot_prod = len(self._get_overlap(j1, j2))
        dot_prod /= (self.profile_lengths[j1] ** self.alpha)
        dot_prod /= (self.profile_lengths[j2] ** (1 - self.alpha))
        return dot_prod

    def _build_profile_lengths(self):
        '''
        2 norm profile length
        '''
        self.profile_lengths = {j:np.sqrt(len(self.ratings.ix[j])) for j in self.items}

    def _build_similarity_cache(self):

        '''
        similarity cache
        '''
        count = 0
        for j1 in self.items:
            for j2 in self.items:
                if j1 != j2 and j2 not in self.sim_cache[j1]:
                    sim_j1_j2 = self._asym_cosine(j1, j2)
                    self.sim_cache[j1][j2] = sim_j1_j2
                    self.sim_cache[j2][j1] = sim_j1_j2
            if count % 10 == 0:
                print ("Processed song {} ({})".format(j1, count))
            count += 1


    def _score(self, user, item):
        '''
        Helper function to calculate the score for item by user.
        '''
        print(self.sim_cache[item])
        "user wanted"
        peers = [i for i in reversed(sorted(self.sim_cache[item], key=operator.itemgetter(1)))]
        profile_length_item = self.profile_lengths[item]
        rating = 0
        
        for peer in peers:
            sim = self.sim_cache[item][peer]
            if sim > 0 and item in list(self.ratings[peer].index):
                rating += (sim ** self.q)
                
        return 0 if profile_length_item == 0 else (rating / (profile_length_item ** (2-2*self.beta)))

    def fit(self, ratings, *args, **kwargs):
        '''
        
        Params:
            
        '''
        self.ratings = ratings.set_index(['song', 'user'])
        self.items = list(ratings['song'].drop_duplicates())
        self._build_profile_lengths()
        self._build_similarity_cache()

    def recommend(self, user, n=None, candidates=None, ratings=None):
        '''
        Should return ordered data frame with items and score. 
        Params:
            user - user to recommend for
            n - number of items to recommend
            candidates - items to choose from
            ratings - no longer needed, it stays none
        '''
        start = time.time()
        # n is None or zero, return DataFrame with an empty item column
        if not n:
            return pd.DataFrame({'item': []})
    


        candidates_so_far = 0
        candidates = self.test_set_candidates[user][0]
        n = min([len(self.test_set_candidates[user][1]), n, len(candidates)])

        # Initialize scores
        scores = {i:0.0 for i in candidates}
        # for each candidate, populate the scores dictionary
        #TODO
        for candidate in candidates:
            scores[candidate] = self._score(user, candidate)

    
        
        # Turn result into data frame
        df = pd.DataFrame.from_dict(scores, orient='index', columns=['score'])

        # Retain n largest scoring rows (nlargest)
        df = df.nlargest(n, 'score')

        # Sort by score (sort_values)
        df = df.sort_values(by=['score'], ascending=False)

        df = df.reset_index()

        print("Processed user", user, self.processed_users + 1, "in", time.time()-start)
        self.processed_users += 1

        # return data frame
        return df.rename(index=str, columns={"index": "item"})

    def get_params(self, deep=True):
        return {'beta': self.beta,
                'alpha': self.alpha,
                'q': self.q,
                'test_set_candidates':self.test_set_candidates}