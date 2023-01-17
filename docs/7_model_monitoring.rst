Monitoring
==========
Monitoring of Kubernetes cluster is carried out with Prometheus & Grafana, 
that allows to connect real-time information about the service and see shifts in data and predictions by metrics.
The model predictive performance is monitored to potentially invoke a new iteration in the ML process.


Monitoring of the status of the k8s cluster 
-------------------------------------------
Prometheus shows overall cluster CPU / Memory / Filesystem usage as well as individual pod, containers, systemd services statistics. 


Monitoring HTTP service metrics
--------------------------------

FastAPI Metrics are connected to Prometheus via starlette_exporter.PrometheusMiddleware.
To expose a metrics endpoint to Prometheus, used the HTTP handler handle_metrics at path /metrics.

starlette_exporter collects basic metrics for FastAPI based applications:

    - starlette_requests_total: a counter representing the total requests
    - starlette_request_duration_seconds: a histogram representing the distribution of request response times
    - starlette_requests_in_progress: a gauge that keeps track of how many concurrent requests are being processed

Metrics include labels for the HTTP method, the path, and the response status code.

Labels and  FastAPI metrics endpoint are added to Kubernetes manifest for service discovery
Here is a link to ServiceMonitor yml file: 
https://github.com/Yanina-Kutovaya/RecSys-retail/blob/main/k8s/monitoring-recsys-retail.yml

The Grafana dashboard is customizable for metrics visualization.


Adding custom metrics for the model to monitoring
----------------------------------------------

Prometheus Python Client provides instruments for collection of different metric types:

    - Counter: a cumulative metric that represents a single monotonically increasing counter whose
     value can only increase or be reset to zero on restart.
    - Gauge: a metric that represents a single numerical value that can arbitrarily go up and down.
    - Histogram: samples observations (usually things like request durations or response sizes) 
    and counts them in configurable buckets. It also provides a sum of all observed values.
    - Summary: samples observations. While it also provides a total count of observations and a sum of all observed values, 
    it calculates configurable quantiles over a sliding time window.
    - Info: any non-numeric information

Model Metrics:

- distribution of input features
- the number of unknown categories
- distribution of predictions (to detect shifts)
- prediction time distribution
- the number of anomalies

Current application monitors the number new customers and the number of predictions made.