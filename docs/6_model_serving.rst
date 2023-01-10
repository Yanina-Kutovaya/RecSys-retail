Model serving
==============

REST API
---------

The validated model is deployed to a target environment to serve predictions. 
This deployment is a microservice with a REST API to serve online predictions.

REST API for the model on python could be found here:

- https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/service/main.py


Github actions CI/CD
---------------------

**Testing:** https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/.github/workflows/python.yml

- Check PEP8 compliance with Black
- Static code analysis with mypy
- Unit tests with pytest: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/tests/test_data_validation.py
- Integration tests with pytest: https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/service/test_main.py  
 
**Docker containerization and its publishing in the registry** https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/.github/workflows/docker.yml
 

Microservice  deployment 
------------------------

**Docker:** https://github.com/Yanina-Kutovaya/RecSys-retail/tree/main/docker

- recsys_retail_train - to build a container just for training of the model
- recsys_retail - to build two types of containers for microservice
- traefik - to use reverse proxy server Traefik for microservice

**Docker-compose**

- To deploy a microservice in development stage: 
    - https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/docker-compose.dev.yml 

- To deploy a microservice in production stage:
    - https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/docker-compose.prod.yml 
 
 
**Kubernetes** 

Prerequisites:

- install Helm: 
    - curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

- installing the NGINX Ingress Controller with a Let's Encrypt® certificates manager
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


k8s manifests: https://github.com/Yanina-Kutovaya/RecSys-retail/tree/main/k8s

- To run a single job - model training:
    - recsys-retail-train-job.yml

- To deploy a microservice in k8s in namespace recsys:
    - recsys-retail-deployment.yml 
    - recsys-retail-service.yml
    - recsys-retail-ingress.yml 
    - prom_rbac.yml
    - production-issuer.yml

- To set up monitoring system with Prometheus and Grafana in namespace monitoring:
    - monitoring-recsys-retail.yml

- To set up monitoring system with Beat, Elasticsearch and Kibana:
    - beats.yml
    - elastic.yml
    - kibana.yml