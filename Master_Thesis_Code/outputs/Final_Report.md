# FINAL REPORT: Multi-Agent Churn Mitigation System

## 1. Executive Summary

This report presents the results of a multi-agent AI system designed to reduce customer churn in the telecommunications sector. The system combines ML-based churn prediction, K-Means customer segmentation, and a multi-agent strategy generation pipeline built on the CrewAI framework.

**Pipeline completed in 171.17s.**

## 2. Pipeline Workflow

1. **Data Loading & EDA**: Load Telco Churn dataset (7,043 customers, 21 features). Generate exploratory charts.
2. **ML Baselines**: Train Logistic Regression and Random Forest to establish churn prediction benchmarks.
3. **Single-Agent Baseline**: A junior analyst agent reads data and proposes basic retention strategies.
4. **Multi-Agent Pipeline**:
   - Data Architect selects numeric features for clustering.
   - CAO runs 20-trial K-Means optimization + GMM comparison.
   - Strategist designs retention campaigns per cluster.
   - Manager (CFO) audits campaigns for financial viability.
5. **Adversarial Loop**: Strategist and Manager iterate up to 3 rounds until approval.
6. **Reporting**: Generate all charts, metrics, and final artifacts.

## 3. Customer Segments (Clusters)

| Cluster | Size | Avg Tenure | Avg Monthly | Avg Total | Churn Rate | Risk |
|---|---|---|---|---|---|---|
| 0 | 1,695 | 10.1 mo | $32 | $307 | 24.8% | Medium |
| 1 | 1,914 | 59.5 mo | $93 | $5,538 | 15.3% | Medium |
| 2 | 2,278 | 15.5 mo | $81 | $1,255 | 48.2% | Very High |
| 3 | 1,156 | 53.4 mo | $34 | $1,810 | 5.0% | Low |

### Cluster Descriptions

**Cluster 0** (n=1,695, Churn: 24.8%, Risk: Medium)

Average tenure: 10.1 months, Monthly: $32, Total: $307. Moderate churn (24.8%). Standard retention programs and periodic engagement should suffice.

**Cluster 1** (n=1,914, Churn: 15.3%, Risk: Medium)

Average tenure: 59.5 months, Monthly: $93, Total: $5,538. Moderate churn (15.3%). Standard retention programs and periodic engagement should suffice.

**Cluster 2** (n=2,278, Churn: 48.2%, Risk: Very High)

Average tenure: 15.5 months, Monthly: $81, Total: $1,255. Highest-risk segment with 48.2% churn. Requires aggressive retention: targeted discounts, contract migration offers, priority support.

**Cluster 3** (n=1,156, Churn: 5.0%, Risk: Low)

Average tenure: 53.4 months, Monthly: $34, Total: $1,810. Low churn (5.0%). Highly loyal segment. Avoid unnecessary discounts — VIP status and non-monetary rewards are more appropriate.

## 4. Approved Campaigns

| Cluster | Persona | OfferType | Discount% | Duration | EstimatedROI% |
|---|---|---|---|---|---|
| 0 | HighRisk | Premium | 30 | 12 | 120% |
| 0 | HighRisk | Standard | 20 | 6 | 90% |
| 1 | ModerateRisk | Premium | 15 | 12 | 100% |
| 1 | ModerateRisk | Standard | 10 | 6 | 80% |
| 2 | HighRisk | Premium | 40 | 12 | 150% |
| 2 | HighRisk | Standard | 30 | 6 | 120% |
| 3 | Loyal | Premium | 0 | 24 | 50% |
| 3 | Loyal | Standard | 0 | 12 | 30% |

### Key Observations

- Cluster 0 has moderate churn (24.8%). Standard retention programs are sufficient.
- Cluster 3 has low churn (5.0%). Minimal or zero discounts are applied to preserve budget.
- Cluster 2 has the highest churn risk (48.2%). Discounts are proportional to churn risk.
- Cluster 1 has moderate churn (15.3%). Standard retention programs are sufficient.
- All ROI estimates are positive: every campaign is designed to generate more revenue than it costs.

## 5. Adversarial Loop Outcome

- **Total loops**: 2
- **Manager (Loop 1)**: REVISION REQUESTED
- **Manager (Loop 2)**: APPROVED

## 6. ML Baseline Summary

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 0.7991 | 0.5916 |
| Random Forest | 0.7906 | 0.559 |

ML models predict churn with ~80% accuracy but cannot prescribe actions. The multi-agent system bridges this gap by converting predictions into actionable campaigns.

## 7. Generated Figures

| Figure | File | Description | Status |
|---|---|---|---|
| Fig 1.1 | Thesis_Pipeline_Architecture.png | Multi-Agent System Architecture | OK |
| Fig 1.2 | Fig_Preprocessing_Pipeline.png | Data Preprocessing Pipeline | OK |
| Fig 2.1 | Fig_EDA_1_Churn_Donut.png | Churn Distribution | OK |
| Fig 2.2a | Fig_EDA_2a_tenure.png | tenure Distribution by Churn | OK |
| Fig 2.2b | Fig_EDA_2b_MonthlyCharges.png | MonthlyCharges Distribution by Churn | OK |
| Fig 2.2c | Fig_EDA_2c_TotalCharges.png | TotalCharges Distribution by Churn | OK |
| Fig 2.2d | Fig_EDA_2d_Contract.png | Contract Distribution by Churn | OK |
| Fig 2.2e | Fig_EDA_2e_PaymentMethod.png | PaymentMethod Distribution by Churn | OK |
| Fig 2.3 | Fig_EDA_3_Correlation.png | Feature Correlation Matrix | OK |
| Fig 2.4 | Fig_EDA_4_Churn_by_Category.png | Churn Rate by Categorical Feature | OK |
| Fig 3.1 | Fig_ML_Confusion_Matrix.png | ML Confusion Matrices | OK |
| Fig 3.2 | Fig_ML_ROC_Curves.png | ML ROC Curves | OK |
| Fig 3.3 | Fig_ML_Feature_Importance.png | Random Forest Feature Importance | OK |
| Fig 3.4 | Fig_ML_Metrics_Table.png | ML Metrics Comparison Table | OK |
| Fig 4.1 | Fig_Optimization_Process.png | Clustering Optimization Process | OK |
| Fig 4.2 | Fig_Elbow_Method.png | Elbow Method for K Selection | OK |
| Fig 4.3 | Fig_PCA_Clusters.png | PCA Cluster Visualization | OK |
| Fig 4.4 | Fig_Cluster_Radar.png | Cluster Profiling Radar Chart | OK |
| Fig 4.5 | Fig_Cluster_Churn_Analysis.png | Cluster Size & Churn Rate | OK |
| Fig 5.1 | Fig_Comp_Performance.png | Performance Comparison | OK |
| Fig 5.2 | Fig_Comp_Cost.png | Token Cost Comparison | OK |
| Fig 5.3 | Fig_Agent_Token_Usage.png | Agent Token Usage Breakdown | OK |

