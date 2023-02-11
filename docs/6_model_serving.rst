Model serving
==============


REST API
---------

The validated model is deployed to a target environment to serve predictions. 
This deployment is a microservice with a REST API to serve online predictions.
Every 100 predictions are saved in Yandex Object Storage, recsys-retail-model-output bucket for futher monitoring of model quality.

REST API for the model on python could be found here:

- https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/service/main.py


Github actions CI/CD
---------------------

**Testing:** https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/.github/workflows/python.yml

- Check PEP8 compliance with Black
- Static code analysis with mypy
- Unit tests with pytest: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/tests/test_data_validation.py
- Integration tests with pytest: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/service/test_main.py  
 
**Docker containerization and its publishing in the registry:** https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/.github/workflows/docker.yml
 
- Trained model which will be deployed in production is saved in Model registry in Yandex Object Storage (YC S3) for further use in A/B tests.
- Features generated on all the intermediate steps are stored in Feature store in YC S3 for further re-use and model analysis.


Microservice  deployment 
------------------------

The application is deployed on Kubernetes cluster. Docker и Docker Compose deployment is also available.

**Kubernetes** 

Prerequisites:

- install Helm: 
    - curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

- install the NGINX Ingress Controller with a Let's Encrypt® certificates manager
    - helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    - helm repo update
    - kubectl create namespace nginx
    - helm install ingress-nginx  --namespace nginx ingress-nginx/ingress-nginx
    - kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.6.1/cert-manager.yaml

- install main kube-prometheus-stack - Prometheus & Grafana 
    - helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    - helm repo update
    - kubectl create namespace monitoring
    - helm install main prometheus-community/kube-prometheus-stack -n monitoring

- install airflow bitnami/airflow with git syncronization to read DAGs from git repository
    - helm install airflow bitnami/airflow \
        - --set git.dags.enabled=true \
        - --set git.dags.repositories[0].repository=https://github.com/Yanina-Kutovaya/RecSys-retail.git \
        - --set git.dags.repositories[0].branch=main \
        - --set git.dags.repositories[0].name=my_dags \
        - --set git.dags.repositories[0].path=dags/ \
        - --set airflow.baseUrl=http://127.0.0.1:8080 \
        - -n airflow


k8s manifests: https://github.com/Yanina-Kutovaya/RecSys-retail/tree/main/k8s

- to run a single job - model training:
    - recsys-retail-train-job.yml

- to deploy a microservice in k8s in namespace recsys:
    - secret.yml
    - recsys-retail-deployment.yml 
    - recsys-retail-service.yml
    - recsys-retail-ingress.yml 
    - prom_rbac.yml
    - production-issuer.yml

- to set up monitoring system with Prometheus and Grafana in namespace monitoring:
    - monitoring-recsys-retail.yml


**Docker:** https://github.com/Yanina-Kutovaya/RecSys-retail/tree/main/docker

- recsys_retail_train - to build a container just for training of the model without model serving
- recsys_retail - to build two types of containers for microservice:
    - container for model serving where the model is trained outside of the container
    - container where on the first stage the model is trained, the second stage is model serving

- traefik - to use a reverse proxy server Traefik for microservice

**Docker-compose:**

- to deploy a microservice in development stage: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/docker-compose.dev.yml 
- to deploy a microservice in production stage: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/docker-compose.prod.yml 