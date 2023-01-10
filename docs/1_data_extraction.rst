# Data extraction: 

Currently, data extraction is outside of the scope of this project. 
Required data is loaded from cloud-based S3 object storage - Yandex Object storage.

It is assumed that stored data is checked for schema skews, which include receiving unexpected features, not receiving all the expected features, or receiving features with unexpected values.
It is also assumed that stored data is checked for values skews - significant changes in the statistical properties of data.

The data is loaded to the model by [load_data](https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/src/recsys_retail/data/make_dataset.py) function. 