# ERP
For DATA72002 Extended Research Project

## Project Code Structure

The project code is organized into several folders. Below is an overview of the structure:

### 1. Data Collection
- **1.1 Stock Selection.py**: Script for selecting the stocks to be analyzed.

### 2. Data Preprocessing
- **2.1 Week-Date-Year mapping table.py**: Script for mapping dates to weeks and years.
- **2.2 Process stock return and text, data merge.py**: Script for processing stock returns and text data.

### 3. EDA (Exploratory Data Analysis)
- **3.1 stock return data visualizations.py**: Script for visualizing stock return data.
- **3.2 headline data visualizations.py**: Script for visualizing headline data.

### 4. Modelling
- **4.1 Pre-trained LLMs: get sentiment score**
  - **4.1.1 BERT.py**: Script for obtaining sentiment scores using BERT.
  - **4.1.2 RoBERTa.py**: Script for obtaining sentiment scores using RoBERTa.
  - **4.1.3 DistilRoBERTa.py**: Script for obtaining sentiment scores using DistilRoBERTa.
  - **4.1.4 FinBERT.py**: Script for obtaining sentiment scores using FinBERT.
  - **4.1.5 DistilBERT.py**: Script for obtaining sentiment scores using DistilBERT.
  
- **4.2 Pre-trained LLMs: From sentiment score to pnret**
  - Script for transforming sentiment scores into return predictions 

- **4.3 Fine-tuned LLMs: get pnret**
  - **4.3.1 BERT.py**: Script for fine-tuning BERT and predicting pnret.
  - **4.3.2 RoBERTa.py**: Script for fine-tuning RoBERTa and predicting pnret.
  - **4.3.3 DistilRoBERTa.py**: Script for fine-tuning DistilRoBERTa and predicting pnret.
  - **4.3.4 FinBERT.py**: Script for fine-tuning FinBERT and predicting pnret.
  - **4.3.5 DistilBERT.py**: Script for fine-tuning DistilBERT and predicting pnret.

- **4.4 after pnret.py**: Script for further analysis after obtaining return predictions.

### 5. Result Illustrations
- **5.1 out-of-sample prediction MSE.py**: Script for calculating MSE of out-of-sample predictions.
- **5.2 out-of-sample prediction accuracy.py**: Script for calculating accuracy of out-of-sample predictions.
- **5.3 Sharpe ratio.py**: Script for calculating the Sharpe ratio.
- **5.4 transaction cost.py**: Script for analyzing transaction costs.
- **5.5 confusion matrix.py**: Script for generating the confusion matrix for each LLM.




## Packages Used

The following Python packages were used in this project:

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical computing, arrays, and matrices.
- **matplotlib**: Data visualization.
- **seaborn**: Statistical data visualization.
- **scikit-learn (sklearn)**: Machine learning tools.
- **os**: Operating system interface.
- **nltk**: Natural language processing.
- **wordcloud**: Generate word clouds from text.
- **joblib**: Model serialization and parallel processing.
- **torch (PyTorch)**: Machine learning and deep learning.
- **transformers**: Pre-trained NLP models.
- **tabulate**: Display tabular data in text form.
