# Fraudulent Jobs Detector with Sentence Transformers (Paraphrase-mpnet-base-v2)

## Overview

This project provides a modular, production-ready pipeline and API for detecting fraudulent job postings using advanced sentence embeddings. It leverages the `paraphrase-mpnet-base-v2` model from Sentence Transformers to generate rich semantic representations of job descriptions and related fields, enabling more nuanced and effective fraud detection.

## Key Features & Strengths

- **State-of-the-art Sentence Embeddings:** Utilizes the `paraphrase-mpnet-base-v2` model to capture deep semantic meaning from job posting text, going far beyond traditional keyword or bag-of-words approaches.
- **Advanced Data Cleaning:** Modular and extensible cleaning pipeline, including optional spell checking, special character handling, and feature selection.
- **Flexible Embedding Pipeline:** Embeddings can be reduced in dimensionality using PCA for efficiency, and the embedding process is configurable for both training and inference.
- **Two-layer Fraud Detection Architecture:** Supports both high-precision and high-recall detection modes, allowing for flexible deployment depending on business needs.
- **Production-ready API:** Flask app served via Waitress for scalable, production-grade inference.
- **Easy Integration:** Simple REST API for fraud prediction, ready to be integrated into job boards or HR platforms.
- **Reproducible Environment:** Dockerfile and requirements.txt provided for seamless setup and deployment.

## ⚠️ Important Note on Model Bias

> **Warning:**  
> The model was trained on a dataset that is known to be somewhat biased. As a result, predictions may reflect these biases and should not be considered fully objective or fair in all contexts. The main focus of this project is to explore the power of sentence embeddings for fraud detection, not to provide a production-ready, bias-free classifier. Please use the results with caution and consider further retraining or bias mitigation for critical applications.

## Project Structure

- `source/classes/`: Modular classes for cleaning, embedding, feature selection, and more.
- `source/training_pipeline.py`: End-to-end training pipeline.
- `source/flask_app/`: Flask API and inference logic.
- `models/`: Trained model artifacts.
- `datasets/`: Training data (e.g., `emscad_without_tags.csv`).

## Getting Started

### Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies.

### Quickstart (Docker)

1. **Build the Docker image:**
   ```bash
   docker build -t fraud-jobs-detector .
   ```
2. **Run the container:**
   ```bash
   docker run -p 5000:5000 fraud-jobs-detector
   ```
3. **Access the API:**  
   Send a POST request to `http://localhost:5000/predict` with a job posting JSON.

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API:
   ```bash
   cd source/flask_app
   python app.py
   ```

## API Usage

- **Endpoint:** `POST /predict`
- **Input:** JSON object with job posting fields.
- **Output:**  
  - `fraudulent_prediction`: 0 (legit) or 1 (fraud)
  - `fraud_probability`: Probability score

## Focus on Sentence Embeddings

The core innovation of this project is the use of sentence embeddings to represent job postings. This allows the model to understand context, semantics, and subtle cues in text that traditional methods miss. The pipeline is designed to make it easy to experiment with different embedding strategies and feature sets.

# Advanced Insights and Best Practices

## Why Sentence Embeddings?

Traditional fraud detection models often rely on keyword matching or simple statistical features, which can miss subtle cues and context present in job postings. By leveraging sentence embeddings, this project captures the semantic meaning of entire sentences and paragraphs, allowing the model to:
- Detect nuanced patterns and relationships in text.
- Generalize better to new, unseen job postings.
- Reduce the need for extensive manual feature engineering.

## Model Interpretability and Monitoring

While the current model is highly effective, it is important to monitor its predictions in production:
- **Model Drift:** Regularly retrain or fine-tune the model as new types of job postings and fraud patterns emerge.
- **Bias Auditing:** Periodically audit predictions for bias, especially if deploying in sensitive or high-stakes environments.
- **Explainability:** Consider integrating tools like LIME or SHAP to provide explanations for individual predictions, increasing trust and transparency.

## Security and Privacy Considerations

- **Data Privacy:** Ensure that any job posting data used for inference or retraining is handled in compliance with relevant data protection regulations (e.g., GDPR).
- **API Security:** If deploying the API publicly, implement authentication, rate limiting, and input validation to prevent misuse.

## Performance and Scalability

- **Batch Inference:** For high-throughput scenarios, adapt the inference pipeline to process batches of job postings efficiently.
- **Horizontal Scaling:** The Dockerized API can be scaled horizontally using orchestration tools like Kubernetes or Docker Compose.
- **Resource Optimization:** Use PCA or similar techniques to reduce embedding dimensionality and speed up inference.

## Customization and Extension

- **Plug-and-Play Embeddings:** Swap out the sentence transformer model for other variants (e.g., multilingual models) with minimal code changes.
- **Feature Engineering:** Easily add new structured or unstructured features to the pipeline using the modular class design.
- **Model Upgrades:** Experiment with other classifiers (e.g., XGBoost, LightGBM, deep learning models) by updating the training pipeline.

## Example API Request

```json
{
  "title": "Senior Data Scientist",
  "description": "We are looking for a data scientist to join our team...",
  "requirements": "PhD in Computer Science or related field...",
  "benefits": "Health insurance, 401k, remote work options...",
  "type_of_contract": "Full-time",
  "type_of_position": "Permanent",
  "required_education": "PhD",
  "nature_of_company": "Tech",
  "nature_of_job": "Research and development",
  "company_profile_and_description": "A leading AI company...",
  "function": "Data Science",
  "company_profile": "Innovative, global, fast-growing"
}
```

## Example API Response

```json
{
  "fraudulent_prediction": 0,
  "fraud_probability": 0.07
}
```

## Acknowledgements

- Built with the support of the open-source community and the maintainers of Sentence Transformers, scikit-learn, and imbalanced-learn.
- Special thanks to contributors and reviewers who provided feedback and improvements.

## Citation

If you use this project in your research or product, please consider citing it or referencing the repository.

---

*This project demonstrates the power of modern NLP and machine learning for real-world risk mitigation. For further collaboration, consulting, or enterprise integration, please contact the maintainer or open an issue.*

## License

This project is for research and educational purposes. Please review the dataset and model licenses before commercial use.

---

# Technical Details

## Data Pipeline

1. **Preliminary Cleaning:**
   - Removes duplicates and repetitive phrases (e.g., in `nature_of_job`).
   - Handles missing values and standardizes text fields.
2. **Advanced Cleaning:**
   - Unified advanced cleaning logic for all relevant textual columns.
   - Optional spell checking (modular, can be toggled).
   - Special character handling: replaces with spaces to preserve token separation.
3. **Feature Engineering:**
   - Dedicated handling for salary-related and other structured features.
   - FeatureSelector class for modular feature concatenation and manipulation.

## Embedding Pipeline

- **Model:** Uses `sentence-transformers/paraphrase-mpnet-base-v2` for generating embeddings.
- **Dimensionality Reduction:** Optional PCA to reduce embedding size for efficiency.
- **Inference:** Embedding logic is shared between training and inference for consistency.
- **Missing Data:** Handles missing or null text by substituting zero vectors.

## Model Training

- **Algorithm:** HistGradientBoostingClassifier (from scikit-learn) with class weighting to address class imbalance.
- **Imbalance Handling:** Uses SMOTE for oversampling the minority (fraud) class.
- **Hyperparameters:** Tuned for stability and generalization (e.g., lower learning rate, early stopping).
- **Two-layer Architecture:**
  - Layer 1: High precision, trained on a subset of features.
  - Layer 2: High recall, trained on the full feature set.

## Inference API

- **Framework:** Flask, served via Waitress for production.
- **Endpoint:** `/predict` (POST)
- **Input:** JSON with job posting fields (title, description, requirements, etc.)
- **Output:**
  - `fraudulent_prediction`: 0 (legit) or 1 (fraud)
  - `fraud_probability`: Probability score
- **Error Handling:** Returns error messages in JSON on failure.

## Deployment

- **Dockerized:** Fully containerized for reproducibility and easy deployment.
- **Port:** Default 5000 (configurable via environment variable).
- **Environment Variables:** Supports standard Flask environment variables.

## Extensibility

- **Modular Classes:** Easy to extend or swap out cleaning, embedding, or model logic.
- **Configurable Pipeline:** Embedding, feature selection, and model parameters are easily adjustable.
- **API-first:** Designed for integration into larger systems or as a microservice.

## Limitations & Future Work

- **Bias:** The model is trained on a biased dataset; further work is needed for fairness and generalization.
- **Explainability:** Current model is a black box; future work could add explainable AI components.
- **Dataset Expansion:** Incorporating more diverse and representative data would improve robustness.

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [scikit-learn HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [SMOTE for Imbalanced Learning](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

For questions or contributions, please open an issue or pull request.
