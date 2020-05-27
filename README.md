# Python-Defect-Prediction-Model

## Model

Please see ***train_and_test.py*** for model implementation in Python.

Please see the ***project_handout.pdf*** for full details on the project and model.

## Building Line-level Defect Detection Models

In this project, we have built a defect prediction model for software source code from scratch. We were required to apply deep-learning techniques, e.g., classification, tokenization, embedding, etc., to build more accurate prediction models with the dataset provided.

### Background

Line-level Defect classifiers predict which lines in a file are likely to be buggy.

A typical line-level defect prediction using deep-learning consists of the following steps:

• Data extraction and labeling: Mining buggy and clean lines from a large dataset of software changes (usually GitHub).

• Tokenization and pre-processing: Deep learning algorithms take a vector as input. Since source code is text, it needs to be tokenized and transformed into a vector before being fed to the model.

• Model Building: Using the tokenized data and labels to train a deep learning classifier. Many different classifiers have been shown to work for text input (RNNs and CNNs). Most of these models can be built using TensorFlow.

• Defect Detection: Unlabelled instances (i.e., line of codes or files) are fed to the trained model that will classify them as buggy or clean.

### Evaluation Metrics

Metrics, i.e., P recision, Recall, and F 1, are widely used to measure the performance of defect prediction models.

These metrics rely on four main numbers: true positive, false positive, true negative, and false negative. True positive is the number of predicted defective instances that are truly defective, while false positive is the number of predicted defective ones that are actually not defective. True positive records the number of predicted non- defective instances that are actually defective, while false negative is the number of predicted non-defective instances that are actually defective. F1 is the weighted average of precision and recall.

These methods are threshold-dependent and are not the best to evaluate binary classifiers. In this project, we will also use the Receiver operating characteristic curve (ROC curve) and its associated metric, Area under the ROC curve (AUC) to evaluate our trained models independently from any thresholds. The ROC curve is created by plotting the true positive rate (or recall, see definition above) against the false positive rate at various threshold settings.

## Using TensorFlow to build a simple classification model

Steps include:

1) Loading input data

2) Preprocessing data

3) Training the model

4) Evaluating the model

## Improving the results by using a better deep-learning algorithm

The model trained in part (I) is simple and does not perform very well. In the past few years, many different models to classify text inputs for diverse tasks (content tagging, sentiment analysis, translation, etc.) have been proposed in the literature. In part (II), you will look at the literature and apply a different deep-learning algorithm to do defect prediction. You can, and are encouraged to use or adapt models that have been proposed by other people for other tasks. Please cite your source and provide a link to a paper or/and GitHub repository showing that this algorithm has been applied successfully for text classification, modeling or generation tasks.
