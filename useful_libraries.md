# Useful materials and libraries for ML in Python
Here I save useful libraries that I (may) need when learning ML and competiting. Sources are wide: Kaggle, GitHub, Medium, random pages...  
The list will be updated regularly, though not guarantee.

## Reading large dataset
- Dask
- RAPIDS (GPU-only tasks)
- Datatable
- TFRecords (especially for Images)

## Computer Vision
- ArcFace
- Generalized Mean
- ViT, DeiT
- labelImg (for object localization)

## Tree-based models
- XGBoost
- LightGBM
- CatBoost
- FastGRF
- Vowpal  Wabbit

## Hyperparameter Tuning
- Sklearn GridSearch, RandomSearch
- Hyperas (for Keras)
- Hyperopt
- Autokeras

## Data Visualization
- Matplotlib (Legend never dies)
- Seaborn 
- Bokeh 
- Plotly (Interactive)
- Altair
- Dash (built on Plotly)
- Pandas Profiling
- klib
- Hiplot
- qgrid

## Data Preprocessing
- Albumentations (Image augmentation)
- imgaug
- StandardScaler, RobustScaler, MinMaxScaler (MinMax and Standard do not affect outlier problem)
- Features Reduction/Generation: PCA, SVD, NMF, LDA, UMAP
- Imbalance: SMOTE/MLSMOTE, ImbalanceDataSampler (PyTorch), ADASYN, Siamese, Triplet Loss
- Fast ICA, FeatureAgglomeration
- Synchronized Batch Norm: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

## NLP
- Embeddings: GloVe, fastText, Word2Vec, Poincare: https://github.com/facebookresearch/poincare-embeddings
- For NLP: CountVectorizer, TF-IDF, N-grams, Word embeddings (Word2Vec, GloVe, fastText, Doc2Vec)
- For NLP: lowercase, lemmatization + stemming, stopwords (NLTK)
- Snorkel (NLP): https://github.com/snorkel-team/snorkel
- EDA: https://github.com/jasonwei20/eda_nlp
- NLPAug: https://github.com/makcedward/nlpaug
- HTML parsing: https://www.crummy.com/software/BeautifulSoup/
- Scrapy: https://scrapy.org/

## Cross-Validation
- Leave-P-Out
- Holdout
- KFold, StratifiedKFold, MultilabelStratifiedKFold
- [Optimise Blending Weights](https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0)

## Optimizers
- RAdam
- Ranger
- Lookahead
- Qhoptim
- QHAdam
- AdamW
- LR Scheduler

## Methods
- Stacking, Boosting, Ensembling, Bagging, Pasting
- Pseudo Labelling
- Label Smoothing, Temperature Scaling, Confidence Penalty (for binary classification)
- DeepInsight (non-image data to image data for CNN)
- Quantile Transformer
- Rank Gauss
- Group Normalization

## Others
- Networkx (for Graph)
- Biopython (for bio-related tasks)
- Streamlit: web app
- TabNet (for tabular data)
- [From multilabel to multiclass](https://www.kaggle.com/c/lish-moa/discussion/200992)
- Numba
- shap (model interpretation)
- GPU-accelerated ML: cudf, cuml (in RAPIDS)
