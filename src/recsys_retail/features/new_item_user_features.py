import logging
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ['generate_user_item_features']


def get_user_item_features(
    recommender, data_train_lvl_1: pd.DataFrame
    ) -> pd.DataFrame:

    logging.info('Generating new user-item features...')

    X = data_train_lvl_1.copy()
    user_item_features = X[['user_id', 'item_id']]    

    # 1.Transaction hour    
    X['hour'] = X['trans_time'] // 100
    df = X.groupby(['user_id', 'item_id'])['hour'].median().reset_index()
    df.columns = ['user_id', 'item_id', 'median_sales_hour']
    user_item_features = user_item_features.merge(df, on=['user_id', 'item_id'])
    
    # 2.Transaction weekday
    X['weekday'] = X['day'] % 7
    df = X.groupby(['user_id', 'item_id'])['weekday'].median().reset_index()
    df.columns = ['user_id', 'item_id', 'median_weekday']
    user_item_features = user_item_features.merge(df, on=['user_id', 'item_id'])
    
    # 3.Mean days between purchases
    df = X.groupby('user_id')['day'].nunique().reset_index()
    df['mean_visits_interval'] = (
        X.groupby('user_id')['day'].max() - X.groupby('user_id')['day'].min()
    ) / df['day']
    user_item_features = user_item_features.merge(
        df[['user_id', 'mean_visits_interval']], on=['user_id']
    )

    # 4.Mean check of user basket
    df = X.groupby(['user_id', 'basket_id'])['sales_value'].sum().reset_index()
    df = df.groupby('user_id')['sales_value'].mean().reset_index()
    df.columns = ['user_id', 'mean_check']
    user_item_features = user_item_features.merge(df, on=['user_id'])
    
    # 5.The number of stores which were selling the item
    df = X.groupby(['item_id'])['store_id'].nunique().reset_index()
    df.columns = ['item_id', 'n_stores']
    user_item_features = user_item_features.merge(df, on=['item_id'])
    
    # 6.The number of unique item bought by the user
    df = X.groupby(['user_id'])['item_id'].nunique().reset_index()
    df.columns = ['user_id', 'n_items']
    user_item_features = user_item_features.merge(df, on=['user_id'])
    
    # 7.The number of user transactions
    df = X.groupby(['user_id'])['item_id'].count().reset_index()
    df.columns = ['user_id', 'n_transactions']
    user_item_features = user_item_features.merge(df, on=['user_id'])
    
    # 8.Mean / max / std of the number of unique items in the user basket
    df = X.groupby(['user_id', 'basket_id'])['item_id'].nunique().reset_index()
    df1 = df.groupby('user_id')['item_id'].mean().reset_index()
    df1.columns = ['user_id', 'mean_n_items_basket']
    user_item_features = user_item_features.merge(df1, on=['user_id'])

    df2 = df.groupby('user_id')['item_id'].max().reset_index()
    df2.columns = ['user_id', 'max_n_items_basket']
    user_item_features = user_item_features.merge(df2, on=['user_id'])

    df3 = df.groupby('user_id')['item_id'].std().reset_index()
    df3.columns = ['user_id', 'std_n_items_basket']
    user_item_features = user_item_features.merge(df3, on=['user_id'])    
       
    # 9.Embeddings
    df1, df2 = get_embeddings(recommender, X)
    user_item_features = user_item_features.merge(df1, on=['item_id'])
    user_item_features = user_item_features.merge(df2, on=['user_id'])
    
    return user_item_features


def get_embeddings(recommender, X: pd.DataFrame) -> pd.DataFrame:

    logging.info('Calculating embeddings...')    

     # Items embeddings    
    df1 = recommender.model.item_factors
    n_factors = recommender.model.factors
    ind = list(recommender.id_to_itemid.values())
    df1 = pd.DataFrame(df1, index=ind).reset_index()
    df1.columns = ['item_id'] + ['factor_' + str(i + 1) for i in range(n_factors)]
    
    # Users embeddings
    df2 = recommender.model.user_factors
    ind = list(recommender.id_to_userid.values())
    df2 = pd.DataFrame(df2, index=ind).reset_index()
    df2.columns = ['user_id'] + ['user_factor_' + str(i + 1) for i in range(n_factors)]

    return df1, df2