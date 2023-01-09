# Recommender system - retail

The goal of the project is to develop a recommender system which would on-line recommend 5 products to the user based on the information available about users and products. 

## Metrics: Precision@5

## Model 
### Pre-filter items
The business goals are to promote sales and generate profit for the company.
To meet business goals, we’ll not include into dataset:

- items that have not been sold for the last 12 months
- the most popular items (they will be bought anyway)
- the most unpopular items (nobody will buy them)
- items from the departments with a limited assortiment
- too cheap items (we will not earn on them)
- too expensive_items (they will be bought irrespective of our recommendations).

For our dataset, we select the top 2500 popular items out of 92353 and introduce fake item_id = 999999: if a user has bought an item which is not from top-N he bought an item 999999.

Here is a link to prefilter_items function: 
https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/prefilter.py 


### Two-stage recommender system
Transaction data is broken into 3 parts: 

-- old purchases -- | -- 6 weeks-- | -- 3 weeks–

On the First stage, we utilize -- old purchases -- together with user and item data to build MainRecommender using Alternating Least Square (ALS) matrix factorization in collaborative filtering. Later, MainRecommender selects top N (100-200) items for each user.

Here is a link to MainRecommender: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/recommenders.py 

On the Second stage, we work with -- 6 weeks-- | -- 3 weeks -- data as train - valid datasets, generating new features and adding to them users and items embeddings from MainRecommender 

Here is a link to get_user_item_features function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/new_item_user_features.py 

Users for the Second stage are selected from both period data: -- 6 weeks-- | -- 3 weeks -- .
For each current user, MainRecommender proposes his own top N purchases. For new users MainRecommender proposes N items from overall top purchases.

Here is a link to get_candidates function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/candidates_lvl_2.py 

Generated users, items and new features are used to generate dataset for the Second stage binary classification model. 

Here is a link to get_targets_lvl_2 function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/targets.py 

Here is a link to train_store model function for RecSys-retail: 
https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/scripts/train_save_model.py 

Binary classification are are used to generate 5 recommendations and calculate Precistion@5 metrics 

Here is a link to get_recommendations and precision_at_k functions: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/metrics.py 

### Inference
FastAPI model is used for inference. Here is a link to the code: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/service/main.py 
