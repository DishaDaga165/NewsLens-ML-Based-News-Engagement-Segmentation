# NewsLens — ML-Based News Engagement Segmentation

News platforms serve users with very different consumption patterns. Some readers browse extensively across topics, some consistently engage with a narrow set of categories, while others interact infrequently but show high intent when they do.

NewsLens explores whether these behavioral differences can be identified from real user interaction data and translated into useful reader segments for content personalization.

The project uses the Microsoft MIND news recommendation dataset and builds an unsupervised machine learning pipeline from raw user-news interactions to behavioral personas.


## Approach

The analysis started with 156,965 impression sessions across 50,000 users and 51,282 news articles. Historical reading activity was combined with current click interactions to construct over 1.1 million unique user-article interactions.

User-level features were then engineered across three dimensions:

- Engagement — session frequency, impressions, click-through rate and total articles consumed
- Content affinity — proportion of consumption across news categories such as sports, finance, lifestyle and entertainment
- Content diversity — Shannon entropy to capture how broadly or narrowly a user consumes content

Users with fewer than 10 unique consumed articles were excluded to avoid inferring preferences from sparse histories, leaving 32,820 users for clustering.

Highly skewed engagement variables were log-transformed before standardization. Extremely rare category features were also removed after finding that standardization could disproportionately amplify sparse categories.

PCA was used to reduce 19 behavioral features to 12 principal components while retaining approximately 81% of the variance.

K-Means models were evaluated across K = 2–8 using Elbow and Silhouette analysis. A four-cluster solution was selected by balancing statistical separation, cluster size and product interpretability.

<img width="1894" height="1020" alt="image" src="https://github.com/user-attachments/assets/9d716d9f-487a-42a5-8a3b-e007fc58abde" />


<img width="1692" height="1010" alt="image" src="https://github.com/user-attachments/assets/b79f4204-642a-4a29-92df-1dd8292bc3e6" />

<img width="1658" height="978" alt="image" src="https://github.com/user-attachments/assets/39f07c09-c7cd-4194-aec1-3dcba154d9b4" />


## Reader Segments

The final model identified four behavioral segments.

| Segment | Users | Observed behavior | Personalization direction |
|---|---:|---|---|
| Power Explorers | 26.7% | Highest consumption volume and broadest category exploration | Cross-category discovery, trending content and deeper feed experiences |
| Focused News Loyalists | 31.8% | Concentrated consumption around news and sports with lower content diversity | Topic-first feeds and affinity-based notifications |
| Lifestyle Explorers | 32.9% | Broader lifestyle, TV and health interests despite similar engagement levels to focused readers | Discovery-oriented recommendations across lifestyle verticals |
| Selective High-Intent Readers | 8.6% | Low content exposure but substantially higher click-through rate | Fewer, high-confidence recommendations with reduced feed overload |

One interesting result was that Focused News Loyalists and Lifestyle Explorers showed similar overall engagement levels but substantially different content preferences and diversity. This suggests that engagement intensity alone may not be enough to determine an appropriate personalization strategy.

Selective High-Intent Readers were also distinct: despite receiving relatively few impressions, their average CTR was approximately 28%, compared with roughly 5% across the other major segments.

<img width="933" height="690" alt="image" src="https://github.com/user-attachments/assets/e22bc522-67de-4e70-9def-f88ebfe6f1b6" />


<img width="606" height="298" alt="image" src="https://github.com/user-attachments/assets/41988848-6d1c-4a44-8d5d-ef0cc5e13973" />



## Model Validation

The number of clusters was not selected solely from the highest Silhouette Score. Higher values of K provided only marginal improvements while creating increasingly fragmented user segments, so cluster size and interpretability were considered alongside Elbow and Silhouette analysis.

Cluster robustness was tested by rerunning K-Means across seven random initializations. The resulting assignments were highly stable, with a mean Adjusted Rand Index (ARI) of 0.98 and a minimum ARI of 0.96.

Ward hierarchical clustering was also run on a reproducible 2,000-user sample as a secondary benchmark. Its agreement with K-Means was limited (ARI = 0.23), indicating that the underlying behavioral groups overlap and that different clustering objectives produce different boundaries.

An important preprocessing issue also emerged during experimentation. An early model created a one-user cluster even though the user's engagement behavior was not unusual. Investigation showed that an extremely rare content category had been amplified during standardization, producing a category-affinity z-score above 180. Rare category features were subsequently filtered and skewed engagement variables were log-transformed before rebuilding the clustering pipeline.


## Limitations

The MIND dataset represents user behavior within a limited observation window, so the resulting segments should be interpreted as behavioral patterns rather than permanent user identities.

The personalization strategies are product recommendations derived from observed cluster characteristics; the project does not measure their causal impact on engagement. In a production setting, these strategies would need to be validated through online experimentation such as A/B tests.

The clustering also shows relatively low absolute Silhouette Scores, suggesting meaningful overlap between reader behaviors rather than perfectly separated natural groups. The segments are therefore best treated as an actionable behavioral abstraction rather than a definitive classification of users.


Dataset: Microsoft MIND (Microsoft News Dataset)

Methods: Feature Engineering · K-Means · PCA · Hierarchical Clustering · Silhouette Analysis · Elbow Method · Standardization · Cluster Stability · Adjusted Rand Index
