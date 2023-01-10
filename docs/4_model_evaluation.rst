# Model evaluation

The model quality is evaluated on a holdout test set.
The results of binary classification are used to generate 5 recommendations and [precision_at_k](https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/metrics.py) function calculates Precistion@5 metrics.