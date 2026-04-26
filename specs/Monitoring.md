# Monitoring Specification & Alert Thresholds

## 1. Service Health
| Signal            | Metric                          | Warning Threshold          | Critical Threshold         | Action on Trigger                                 | Owner       |
|-------------------|----------------------------------|----------------------------|----------------------------|---------------------------------------------------|-------------|
| Latency           | P95 inference latency (ms)       | >25ms for 5 min            | >30ms for 2 min            | Auto-scale or revert to previous model            | infra team  |
| Error Rate        | HTTP 5xx / prediction failures   | >1% for 5 min              | >3% for 1 min              | Pause canary, rollback to last stable version     | infra team  |
| Throughput        | Requests per second              | <80% of expected           | <50% of expected           | Check upstream load balancer, scale up            | infra team  |
| Memory            | RAM usage (MB)                   | >80% of limit              | >95% of limit              | Restart container, investigate leak               | infra team  |

## 2. Data Drift Signals
| Signal            | Method                           | Warning Threshold          | Critical Threshold         | Action on Trigger                                 | Owner       |
|-------------------|----------------------------------|----------------------------|----------------------------|---------------------------------------------------|-------------|
| Input Drift       | PSI on signal distribution       | PSI > 0.05                 | PSI > 0.1                  | Investigate new data source; schedule retraining  | data team   |
| Feature Missing   | % of samples with missing values | >1%                        | >5%                        | Alert upstream pipeline; fix data contract        | data team   |
| Label Delay       | Time since last labeled sample   | >7 days                    | >14 days                   | Trigger manual labeling pipeline                  | ops team    |

## 3. Model Quality Proxies (ground truth delayed)
| Signal            | Proxy Metric                     | Warning Threshold          | Critical Threshold         | Action on Trigger                                 | Owner       |
|-------------------|----------------------------------|----------------------------|----------------------------|---------------------------------------------------|-------------|
| Confidence Drop   | Average prediction entropy       | >0.6                       | >0.8                       | Run eval on golden set; if accuracy drop >2%, rollback | ml team |
| Output Stability  | % of predictions changing >1 class per week | >10%            | >20%                       | Investigate concept drift; retrain                | ml team     |
| Golden Set Score  | Accuracy on golden set (weekly)  | <0.80                      | <0.75                      | Rollback to last known good version               | ml team     |

## 4. Cost & Resource
| Signal            | Metric                          | Warning Threshold          | Critical Threshold         | Action on Trigger                                 | Owner       |
|-------------------|----------------------------------|----------------------------|----------------------------|---------------------------------------------------|-------------|
| Cost per Request  | Average cost per inference ($)   | >0.01                      | >0.02                      | Optimize model or switch to cheaper tier          | finops team |
| Daily Budget      | Cumulative cost for the day      | >80% of daily budget       | >100% of daily budget      | Enable degradation mode (use fallback model)      | finops team |

## Alert Routing
- All **critical** alerts trigger PagerDuty notification to the owner.
- All **warning** alerts are logged to #alerts Slack channel and reviewed during daily standup.