Data preparation and model training
====================================

Pre-filter items
-----------------
The business goals are to promote sales and generate profit for the company.
To meet business goals, we’ll not include into dataset:

- items that have not been sold for the last 12 months
- the most popular items (they will be bought anyway)
- the most unpopular items (nobody will buy them)
- items from the departments with a limited assortiment
- too cheap items (we will not earn on them)
- too expensive_items (they will be bought irrespective of our recommendations).

For our dataset, we select the top 2500 popular items out of 92353 and introduce fake item_id = 999999: 
if a user has bought an item which is not from top-N he bought an item 999999.

Products set for the service is defined by prefilter_items function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/prefilter.py


Preprocess user and item features 
------------------------------------

- fit_transform_user_features function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/user_features.py
- fit_transform_item_features function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/item_features.py 


Two-stage recommender system
-----------------------------
Transaction data is broken into 3 parts: 

-- old purchases -- | -- 6 weeks -- | -- 3 weeks --

Here is a link to time_split function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/data_time_split.py


**MainRecommender** https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/recommenders.py

On the First stage, we utilize -- old purchases -- together with user and item data to build MainRecommender 
using Alternating Least Square (ALS) matrix factorization in collaborative filtering. 
Later, MainRecommender selects top N (100-200) items for each user.


**Dataset for the Second stage model** 

On the Second stage, we work with -- 6 weeks-- | -- 3 weeks -- data as train - valid datasets, 
get_user_item_features function generates new features and adds to them users and items embeddings from MainRecommender: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/new_item_user_features.py
    
Users for the Second stage are selected by get_candidates function from both periods' data: -- 6 weeks-- | -- 3 weeks -- : https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/candidates_lvl_2.py

For each current user, MainRecommender proposes his own top N purchases. 
For new users MainRecommender proposes N items from overall top purchases.

Selected users, items and new features are used by get_targets_lvl_2 function to generate dataset for the Second stage binary classification model: 
https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/features/targets.py

All the functions involved in train dataset formation for the Second stage model are placed into data_preprocessing_pipeline: 
https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/models/train.py

All the artifacts are saved in Feature registry, some of them will be re-used in inference. 
The function load_inference_artifacts uploads artifacts into preprocess function which generates data for inference based on user id.

- load_inference_artifacts function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/models/load_artifacts.py
- preprocess function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/models/inference_tools.py


**Binary classification model**
  
The model is trained and saved by train_store function: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/scripts/train_save_model.py

