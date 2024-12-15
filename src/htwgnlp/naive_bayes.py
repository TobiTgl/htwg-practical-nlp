"""Naive Bayes classifier for NLP.

This module contains the NaiveBayes class for NLP tasks.

Implementing this module is the 3rd assignment of the course. You can find your tasks by searching for `TODO ASSIGNMENT-3` comments.

Hints:
- Find more information about the Python property decorator [here](https://www.programiz.com/python-programming/property)
- To build the word frequencies, you can use the [Counter](https://docs.python.org/3/library/collections.html#collections.Counter) class from Python's collections module
- you may also find the Python [zip](https://docs.python.org/3/library/functions.html#zip) function useful.
- for prediction, you may find the [intersection](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.intersection.html) method of the pandas Index class useful.

"""

from collections import Counter

import numpy as np
import pandas as pd


class NaiveBayes:
    """Naive Bayes classifier for NLP tasks.

    This class implements a Naive Bayes classifier for NLP tasks.
    It can be used for binary classification tasks.

    Attributes:
        word_probabilities (pd.DataFrame): the word probabilities per class, None before training
        df_freqs (pd.DataFrame): the word frequencies per class, None before training. The index of the DataFrame is the vocabulary.
        log_ratios (pd.Series): the log ratios of the word probabilities, None before training. The index of the Series is the vocabulary.
        logprior (float): the logprior of the model, 0 before training. The index of the Series is the vocabulary.
        alpha (float): the smoothing parameter of the model
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initializes the NaiveBayes class.

        The init method accepts one hyperparameter as an optional argument, the smoothing parameter alpha.

        Args:
            alpha (float, optional): the smoothing parameter. Defaults to 1.0.
        """
        self.alpha = alpha
        self.word_probabilities: pd.DataFrame = None
        self.df_freqs: pd.DataFrame = None
        self.log_ratios: pd.Series = None
        self._logprior: float = 0

    @property
    def logprior(self) -> float:
        """Returns the logprior.

        Returns:
            float: the logprior
        """
        return self._logprior

    @logprior.setter
    def logprior(self, y: np.ndarray) -> None:
        """Sets the logprior.

        Note that `y` must contain both classes.

        Args:
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        class_counts = np.bincount(y.flatten())
        assert (
            len(class_counts) == 2
        ), "y must contain exactly two classes (e.g., 0 and 1)."
        assert class_counts[0] != 0, "y must contain both classes."
        assert class_counts[1] != 0, "y must contain both classes."

        # Calculate logprior
        self._logprior = np.log(class_counts[1] / class_counts[0])

    def _get_word_frequencies(self, X: list[list[str]], y: np.ndarray) -> None:
        """Computes the word frequencies per class.

        For a given list of tokenized text and a numpy array of class labels, the method computes the word frequencies for each class and stores them as a pandas DataFrame in the `df_freqs` attribute.

        In pandas, if a word does not occur in a class, the frequency should be set to 0, and not to NaN. Also make sure that the frequencies are of type int.

        Note that the this implementation of Naive Bayes is designed for binary classification.

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples.
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples.
        """
        class_0_words = []
        class_1_words = []

        for tokens, label in zip(X, y.flatten()):
            if label == 0:
                class_0_words.extend(tokens)
            elif label == 1:
                class_1_words.extend(tokens)
            else:
                raise ValueError("y must contain only two classes: 0 and 1.")

        # Count word frequencies for each class using Counter
        class_0_counts = Counter(class_0_words)
        class_1_counts = Counter(class_1_words)

        # Combine vocabularies from both classes
        vocab = list(set(class_0_counts.keys()).union(set(class_1_counts.keys())))

        # Build a DataFrame with frequencies
        self.df_freqs = pd.DataFrame(
            {
                0: [class_0_counts.get(word, 0) for word in vocab],
                1: [class_1_counts.get(word, 0) for word in vocab],
            },
            index=vocab,
        )

        # Ensure the frequencies are integers
        self.df_freqs = self.df_freqs.astype(int)

    def _get_word_probabilities(self) -> None:
        """Computes the conditional probabilities of a word given a class using Laplacian Smoothing.

        Based on the word frequencies, the method computes the conditional probabilities for a word given its class and stores them in the `word_probabilities` attribute.
        """
        # Compute total word counts per class
        total_class_0 = self.df_freqs[0].sum()
        total_class_1 = self.df_freqs[1].sum()

        # Vocabulary size (number of unique words)
        vocab_size = len(self.df_freqs)

        # Apply Laplacian smoothing to compute probabilities
        self.word_probabilities = pd.DataFrame(
            {
                0: (self.df_freqs[0] + self.alpha)
                / (total_class_0 + self.alpha * vocab_size),
                1: (self.df_freqs[1] + self.alpha)
                / (total_class_1 + self.alpha * vocab_size),
            },
            index=self.df_freqs.index,
        )

    def _get_log_ratios(self) -> None:
        """Computes the log ratio of the conditional probabilities.

        Based on the word probabilities, the method computes the log ratios and stores them in the `log_ratios` attribute.
        """
        # Compute log ratios
        self.log_ratios = np.log(
            self.word_probabilities[1] / self.word_probabilities[0]
        )

    def fit(self, X: list[list[str]], y: np.ndarray) -> None:
        """Fits a Naive Bayes model for the given text samples and labels.

        Before training naive bayes, a couple of assertions are performed to check the validity of the input data:
            - The number of text samples and labels must be equal.
            - y must be a 2-dimensional array.
            - y must be a column vector.

        if all assertions pass, the method calls the Naive Bayes training method is executed.

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        assert len(X) == len(y), "X and y must have the same length."
        assert y.ndim == 2, "y must be a 2-dimensional array."
        assert y.shape[1] == 1, "y must be a column vector."

        self._train_naive_bayes(X, y)

    def _train_naive_bayes(self, X: list[list[str]], y: np.ndarray) -> None:
        """Trains a Naive Bayes model for the given text samples and labels.

        Training is done in four steps:
            - Compute the log prior ratio
            - Compute the word frequencies
            - Compute the word probabilities of a word given a class using Laplacian Smoothing
            - Compute the log ratios

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        self.logprior = y
        self._get_word_frequencies(X, y)
        self._get_word_probabilities()
        self._get_log_ratios()

    def predict(self, X: list[list[str]]) -> np.ndarray:
        """Predicts the class labels for the given text samples.

        The class labels are returned as a column vector, where each entry represents the class label of the corresponding sample.

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples

        Returns:
            np.ndarray: a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        # Prepare an empty list for predictions
        predictions = []

        # Iterate over each tokenized sample
        for tokens in X:
            # Compute the sum of log ratios for the given tokens
            log_sum = self.logprior + sum(
                self.log_ratios.get(token, 0) for token in tokens
            )

            # Predict class based on the sign of log_sum
            predicted_class = 1 if log_sum > 0 else 0
            predictions.append(predicted_class)

        # Return predictions as a column vector
        return np.array(predictions).reshape(-1, 1)

    def predict_prob(self, X: list[list[str]]) -> np.ndarray:
        """Calculates the log likelihoods for the given text samples.

        The class probabilities are returned as a column vector, where each entry represents the probability of the corresponding sample.

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples

        Returns:
            np.ndarray: a numpy array of class probabilities of shape (m, 1), where m is the number of samples
        """
        if self.log_ratios is None or self.logprior is None:
            raise ValueError(
                "The model must be trained before calculating probabilities."
            )

        # Prepare an empty list for log likelihoods
        log_likelihoods = []

        # Iterate over each tokenized sample
        for tokens in X:
            # Compute the log likelihood sum for the given tokens
            log_sum = self.logprior + sum(
                self.log_ratios.get(token, 0) for token in tokens
            )
            log_likelihoods.append(log_sum)

        # Return log likelihoods as a column vector
        return np.array(log_likelihoods).reshape(-1, 1)

    def predict_single(self, x: list[str]) -> float:
        """Calculates the log likelihood for a single text sample.

        Words that are not in the vocabulary are ignored.

        Args:
            x (list[str]): a tokenized text sample

        Returns:
            float: the log likelihood of the text sample
        """
        # Compute the sum of log ratios for the given tokens
        log_sum = self.logprior + sum(self.log_ratios.get(token, 0) for token in x)

        return log_sum
