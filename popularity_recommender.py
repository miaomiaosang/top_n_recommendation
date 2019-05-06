from lenskit.algorithms import Recommender
import numpy as np
import pandas as pd
import time

class PopularityRecommender(Recommender):

    def __init__(self, test_set_candidates=None):
        self.popdf = None
        self.processed_users = 0
        self.test_set_candidates = test_set_candidates

    def fit(self, ratings, *args, **kwargs):
        '''
        Building popularity DataFrame, popdf, with columns: ['item': str, 'rank': int]
        Params:
            ratings - DataFrame with columns: ['user': str, 'song': str, 'play_count': int]
        '''
        self.popdf = pd.DataFrame(ratings.groupby('song').size()).sort_values(by=0, ascending=False)

    def recommend(self, user, n=None, candidates=None, ratings=None):
        '''
        Should return ordered data frame with items and score. 
        Params:
            user - user to recommend for
            n - number of items to recommend
            candidates - items to choose from (stays None)
            ratings - to generate candidates dynamically (stays None)
        '''
        start = time.time()
        # n is None or zero, return DataFrame with an empty item column
        if not n:
            return pd.DataFrame({'item': []})

        # candidates = self.test_set_candidates[user][0]
        # n = min([len(self.test_set_candidates[user][1]), n, len(candidates)]) # it may happen that number of candidates left to recommend are less than N
        # Fabio may have followed the following strategy though:
        candidates = list(self.popdf.index)
        n = 500

        # Initialize scores
        scores = {i:0.0 for i in candidates}

        candidates_so_far = 0
        # for each candidate
        for potential_candidate in self.popdf.iterrows():
            # Score the candidate for the user
            if potential_candidate[0] in candidates:
                scores[potential_candidate[0]] = potential_candidate[1][0]
                candidates_so_far += 1
            if candidates_so_far >= n:
                break

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
        return {'test_set_candidates': self.test_set_candidates}