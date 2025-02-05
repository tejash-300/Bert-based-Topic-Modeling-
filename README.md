# **BERT Topic Modeling**

## **Overview**
This project utilizes **BERT (Bidirectional Encoder Representations from Transformers)** for topic modeling, allowing the extraction of coherent topics from large unstructured datasets. Unlike traditional methods like Latent Dirichlet Allocation (LDA), this approach leverages BERT embeddings to capture semantic relationships in text, resulting in more meaningful and accurate topic clusters.

## **Features**
- **BERT-Based Embeddings**: Leverages BERT to generate contextualized embeddings for input text.
- **Dimensionality Reduction**: Applies techniques like **t-SNE** or **UMAP** to reduce embedding dimensions for clustering.
- **Clustering**: Utilizes clustering algorithms such as **K-Means** or **HDBSCAN** to group text data into topics.
- **Topic Representation**: Extracts representative keywords for each topic to summarize them effectively.
- **Interactive Visualization**: Creates visual plots to explore clusters and their distribution.


## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - `transformers` (Hugging Face): For BERT embeddings.
  - `scikit-learn`: For clustering and dimensionality reduction.
  - `matplotlib` and `seaborn`: For data visualization.
  - `UMAP` or `t-SNE`: For dimensionality reduction.
  - `pandas` and `numpy`: For data manipulation and analysis.

## **Setup and Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tejash-300/bert-topic-modeling.git
   cd bert-topic-modeling
   ```

2. **Install Dependencies**:
   Ensure Python is installed. Then, install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pretrained BERT Model**:
   The project uses Hugging Face's BERT model. Download it:
   ```bash
   from transformers import AutoTokenizer, AutoModel
   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   model = AutoModel.from_pretrained("bert-base-uncased")
   ```


## **Usage**

1. **Prepare the Input Data**:
   Ensure your dataset is in a CSV or TXT format with a column containing the text to analyze.

2. **Run the Notebook**:
   Open the `bert_topic_modeling.ipynb` file in Jupyter Notebook or Google Colab and execute the cells step-by-step.

3. **Example Input**:
   ```plaintext
   "The quick brown fox jumps over the lazy dog."
   "AI is transforming the field of natural language processing."
   ```

4. **Example Output**:
   ```plaintext
   Topic 1: ['AI', 'transforming', 'natural', 'language', 'processing']
   Topic 2: ['fox', 'quick', 'brown', 'jumps', 'dog']
   ```

5. **Visualization**:
   Explore the topic clusters visually using **t-SNE** or **UMAP** scatter plots.


## **Future Enhancements**
- Integrate support for multilingual topic modeling using `bert-multilingual-cased`.
- Add a web-based interface for easier interaction.
- Automate hyperparameter tuning for clustering algorithms.
- Improve scalability for large datasets using distributed processing.

---

## **Contributing**
Contributions are welcome! Fork the repository and create a pull request for any improvements or feature additions.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contact**
For questions or suggestions, feel free to reach out:
- **Email**: tejashpandey740@gmail.com
- **GitHub**: [tejash-300](https://github.com/tejash-300)
