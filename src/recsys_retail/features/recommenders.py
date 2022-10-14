import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Recommendations from ALS model and ItemItemRecommender
    Input
    -----
    user_item_matrix: pd.DataFrame
        Matrix of interactions user-item
    """

    def __init__(
        self, data, weighting=True, 
        n_factors_ALS=20, 
        regularization_ALS=0.001, 
        iterations_ALS=15, 
        num_threads_ALS=4
        ):
                
        self.n_factors = n_factors_ALS
        self.regularization = regularization_ALS
        self.iterations = iterations_ALS
        self.num_threads = num_threads_ALS

        # Top purchases of each user
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Top purchases of all dataset
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[
            self.overall_top_purchases['item_id'] != 999999
        ]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        # Prepare user_item_matrix and axillary dictionaries for implicit
        self.user_item_matrix = self._prepare_matrix(data)  
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        # Make bm25 weighting
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        # Train ALS model and ItemItemRecommender
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data):
        """Prepares user_item_matrix for implicit"""
        user_item_matrix = pd.pivot_table(
            data,
            index='item_id', columns='user_id', values='quantity',  
            aggfunc='count', fill_value=0
        )
        user_item_matrix = user_item_matrix.astype(float)  

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Prepares axillary dictionaries"""

        userids = user_item_matrix.columns.values
        itemids = user_item_matrix.index.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id


    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Trains ItemItemRecommender, which recommends items from already bought by user"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender
    
    
    def fit(self, user_item_matrix):
        """Trains ALS model"""

        model = AlternatingLeastSquares(
            factors=self.n_factors, regularization=self.regularization,
            iterations=self.iterations, num_threads=self.num_threads
        )
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model
    

    def _update_dict(self, user_id):
        """Updates dictionaries in case of new user / item"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})


    def _get_similar_item(self, item_id):
        """Finds item similar to item_id"""

        # As item is similar to itself, recommends 2 items (N=2)
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        # Takes the 2nd item (not the item_id from this method's argument) 
        top_rec = recs[1][0]

        return self.id_to_itemid[top_rec]


    def _extend_with_top_popular(self, recommendations, N=5):
        """If the number of recommendations is less than N, adds top purchases of all dataset"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations


    def _get_recommendations(self, user, model, N=5):
        """Recommendations from implicit (standard libraries)"""

        self._update_dict(user_id=user)
        res = [
            self.id_to_itemid[rec] for rec in model.recommend(
                userid=self.userid_to_id[user], 
                user_items=csr_matrix(self.user_item_matrix).tocsr(),
                N=N,
                filter_already_liked_items=False,
                #filter_items=[self.itemid_to_id[999999]],
                recalculate_user=True
            )[0]
        ][:N]
        res = self._extend_with_top_popular(res, N=N)
        assert len(res) == N, f'The number of recommendations != {N}, user_id = {user}'

        return res


    def get_als_recommendations(self, user, N=5):
        """Recommendations from implicit (standard libraries)"""

        self._update_dict(user_id=user)        
        return self._get_recommendations(user, model=self.model, N=N)


    def get_own_recommendations(self, user, N=5):
        """Recommendations from the items, which the user has already bought"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)


    def get_similar_items_recommendation(self, user, N=5):
        """Recommendations of items similar to top-N bought by the"""

        top_users_purchases = self.top_purchases[
            self.top_purchases['user_id'] == user
        ].head(N)
        res = top_users_purchases['item_id'].apply(
            lambda x: self._get_similar_item(x)
        ).tolist()
        res = self._extend_with_top_popular(res, N=N)
        assert len(res) == N, f'The number of recommendations != {N}'

        return res


    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        # Find top-N similar users
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        
        # Eliminate the current user from request
        similar_users = similar_users[1:]   

        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))

        res = self._extend_with_top_popular(res, N=N)
        assert len(res) == N, f'The number of recommendations != {N}'

        return res