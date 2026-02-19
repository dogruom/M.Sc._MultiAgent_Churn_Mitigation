# EXPERIMENTAL SETUP & COMPUTE ENVIRONMENT

## Hardware & Software
| Component | Specification |
|---|---|
| OS | Windows 11 |
| Python Version | 3.12.9 |
| Processor | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| RAM (GB) | 15.71 |
| LLM Providers | 3 providers in rotation pool |

## Library Versions
| Library | Version |
|---|---|
| pandas | 3.0.1 |
| scikit-learn | 1.8.0 |
| crewai | 1.9.3 |
| numpy | 2.4.2 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| litellm | 1.75.3 |

## Experimental Parameters
| Parameter | Value |
|---|---|
| Random State | 42 (all stochastic operations) |
| Train/Test Split | 80/20, stratified on Churn |
| Clustering Optimization | 20 Trials (Hybrid Strategy: Business + Random) |
| Clustering Algorithms | K-Means (primary) + GMM (comparison) |
| ML Baselines | Logistic Regression + Random Forest |
| Adversarial Loop | Max 3 iterations |
| Agent Comparison Trials | N_RUNS=10 (mean ± std reported) |
| ML Cross-Validation | 5-Fold Stratified |
| Clustering Stability | 5 seeds (42, 123, 456, 789, 1024) |
| Reproducibility | random_state=42 for all stochastic operations |
| Feature Scaling | StandardScaler |
| Missing Value Imputation | SimpleImputer (median) |
| Token Budget | 10000 tokens/run |

## LLM Provider Pool
| Provider | Model | RPM | RPD | TPM |
|---|---|---|---|---|
| groq | groq/llama-3.3-70b-versatile | 30 | 14400 | 6000 |
| groq_2 | groq/llama-3.3-70b-versatile | 30 | 14400 | 6000 |
| groq_3 | groq/llama-3.3-70b-versatile | 30 | 14400 | 6000 |
