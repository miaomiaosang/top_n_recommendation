from lenskit.algorithms import Recommender
from collections import defaultdict
import numpy as np
import pandas as pd
import time

class UserBasedRecommender(Recommender):

    def __init__(self, beta = 1, alpha = 0, q = 1):
        self.beta = beta
        self.alpha = alpha
        self.q = q
        self.sim_cache = defaultdict(lambda: defaultdict(float))
        self.ratings = None
        self.users = None
        self.profile_lengths = defaultdict(int)

    def _get_overlap(self, u1, u2):
        '''
        Helper function to get overlap items between two users
        '''
        songs_u1 = set(self.ratings.ix[u1].index)
        songs_u2 = set(self.ratings.ix[u2].index)
        return list(songs_u1.intersection(songs_u2))

    def _asym_cosine(self, u1, u2):
        '''
        Helper function to calculate asymmetric cosine similarity between two users u1 and u2.
        '''
        dot_prod = len(self._get_overlap(u1, u2))
        dot_prod /= (self.profile_lengths[u1] ** self.alpha)
        dot_prod /= (self.profile_lengths[u2] ** (1 - self.alpha))
        return dot_prod

    def _build_profile_lengths(self):
        '''
        Helper function to populate profile lengths.
        '''
        self.profile_lengths = {u:np.sqrt(len(self.ratings.ix[u])) for u in self.users}

    def _build_similarity_cache(self):
        '''
        Helper function to populate similarity cache using alpha parameter. Called by fit function.
        '''
        count = 0
        for u1 in self.users:
            for u2 in self.users:
                if u1 != u2 and u2 not in self.sim_cache[u1]:
                    sim_u1_u2 = self._asym_cosine(u1, u2)
                    self.sim_cache[u1][u2] = sim_u1_u2
                    self.sim_cache[u2][u1] = sim_u1_u2
            if count % 10 == 0:
                print ("Processed user {} ({})".format(u1, count))
            count += 1

    def _score(self, user, item):
        '''
        Helper function to calculate the score for item by user.
        '''
        peers = [i[0] for i in reversed(sorted(self.sim_cache[user].items(), key=operator.itemgetter(1)))]
        #TODO q used here to specify how many neighbors to be utilized!
        rating = 0
        total_peers = 0
        for peer in peers:
            sim = self.sim_cache[user][peer]
            if sim > 0 and item in list(self.ratings[peer].index):
                rating += (sim ** self.q)
                total_peers += 1
        return 0 if total_peers == 0 else (rating / (total_peers ** (1-self.beta)))

    def fit(self, ratings, *args, **kwargs):
        '''
        Populate similarity cache between users using asymmetric cosine function.
        Params:
            ratings - DataFrame with columns: ['user': str, 'song': str, 'play_count': int]
        '''
        self.ratings = ratings.set_index(['user', 'song'])
        self.users = list(ratings['user'].drop_duplicates())
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
    
        # Initialize scores
        scores = {i:0.0 for i in candidates}

        candidates_so_far = 0
        candidates = self.test_set_candidates[user][0]
        n = min([len(self.test_set_candidates[user][1]), n, len(candidates)])

        # for each candidate, populate the scores dictionary
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
                'q': self.q}