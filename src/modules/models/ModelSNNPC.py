# imports
import numpy as np
import os
import torch
import snntorch.functional as SF

from snntorch.functional.acc import _prediction_check, _population_code
from snntorch.surrogate import fast_sigmoid
from torch import from_numpy
from torch.utils.data import DataLoader
from modules.dataset.DatasetBAF import DatasetBAF
from modules.networks.csnnpc import CSNNPC
from modules.metrics import evaluate, evaluate_business_constraint, evaluate_fairness

class ModelSNNPC(object):
    """
    Class to create a Spike Neural Network with Population Coding.
    ------------------------------------------------------
    Args:
        num_features (int): number of features
        num_classes (int): number of classes
        population (int): number of neurons in the population
        class_weights (tuple): weights for the classes
        batch_size (int): size of the batch
        betas (tuple): betas for the network
        slope (int): slope for the fast sigmoid
        thresholds (tuple): thresholds for the network
        num_epochs (int): number of epochs
        num_steps (int): number of steps
        adam_betas (tuple): betas for the Adam optimizer
        learning_rate (float): learning rate for the optimizer
        verbose (int): level of verbosity
    """
    def __init__(self, num_features, num_classes, population, class_weights=(0.02, 0.98), batch_size=128, betas=(0.5,0.5,0.5), slope=25, thresholds=(0.2,0.2,0.2), num_epochs=1, num_steps=20, adam_betas=(0.9, 0.999), learning_rate=1e-5, verbose=1):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        self.num_features = num_features
        self.num_classes = num_classes
        self.population = population
        self.batch_size = batch_size 
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.adam_betas = adam_betas
        self.verbose = verbose
        self.betas = betas
        self.slope = slope
        self.thresholds = thresholds
        self.spike_grad = fast_sigmoid(slope=slope)
        self._dtype = torch.float
        self._device = device
        self.network = self._loadnetwork()
        self.class_weights = torch.tensor(class_weights, dtype=self._dtype, device=self._device)
        self._loss_fn = SF.ce_count_loss(weight=self.class_weights, population_code=True, num_classes=2)
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, betas=self.adam_betas)

    
    def _load_data(self, x, y):
        """Load the data."""
        x_np = from_numpy(x.values).float().unsqueeze(1)
        y_np = from_numpy(y.values).int()
        ds = DatasetBAF(x_np, y_np)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return loader

    def _loadnetwork(self):
        """Load the network."""
        network = CSNNPC(self.num_features, self.num_classes, self.population, self.betas, self.spike_grad, self.num_steps, self.thresholds)
        print(network) if self.verbose >= 2 else None
        return network.to(self._device)

    def fit(self, x_train, y_train):
        """
        Fit the model to the training set.
        ------------------------------------------------------
        Args:
            x_train (pd.DataFrame): dataframe with the training features
            y_train (pd.Series): series with the training labels
        """
        self._train_loader = self._load_data(x_train, y_train)
        for epoch in range(self.num_epochs):
            print(f"Epoch - {epoch}") if self.verbose >= 2 else None
            train_batch = iter(self._train_loader)
            for data, targets in train_batch:
                data = data.to(self._device)
                targets = targets.to(self._device, dtype=torch.long)
                self.network.train()
                _, spk_rec, _ = self.network(data)
                loss_val = self._loss_fn(spk_rec, targets)
                self._optimizer.zero_grad()
                loss_val.backward()
                self._optimizer.step()
               

    def predict(self, x_test, y_test):
        """
        Predict the labels of the test set.
        ------------------------------------------------------
        Args:
            x_test (pd.DataFrame): dataframe with the test features
            y_test (pd.Series): series with the test labels
        ------------------------------------------------------
        Returns:
            predictions (np.array): array with the predictions
            test_targets (np.array): array with the true values
        """
        self._test_loader = self._load_data(x_test, y_test)
        predictions = np.array([])
        test_targets = np.array([])
        with torch.no_grad():
            self.network.eval()
            for data, targets in iter(self._test_loader):
                data = data.to(self._device)
                targets = targets.to(self._device, dtype=torch.long)
                _, spk_rec, _ = self.network(data)
                _, _, num_outputs = _prediction_check(spk_rec)
                _, idx = _population_code(spk_rec, self.num_classes, num_outputs).max(1)
                predictions = np.append(predictions, idx.cpu().numpy())
                test_targets = np.append(test_targets, targets.cpu().numpy())
        return predictions, test_targets
    

    def evaluate(self, targets, predicted):
        """Evaluate the model using the confusion matrix and some metrics.
        ------------------------------------------------------ 
        Args:
            targets (list): list of true values
            predicted (list): list of predicted values
        ------------------------------------------------------ 
        Returns:
            cm (np.array): confusion matrix
            tn (int): true negative
            fp (int): false positive
            fn (int): false negative
            tp (int): true positive
            accuracy (float): accuracy of the model
            precision (float): precision of the model
            recall (float): recall of the model
            fpr (float): false positive rate of the model
            tnr (float): true negative rate of the model
            f1_score (float): f1 score of the model
            auc (float): area under the curve of the model
        """
        metrics = evaluate(targets, predicted)
        return metrics
    
    def evaluate_business_constraint(self, y_test, predictions):
        """Evaluate the model using the business constraint of 5% FPR.
        ------------------------------------------------------
        Args:
            x_test (pd.DataFrame): dataframe with the test features
            y_test (pd.Series): series with the test labels
            predictions (np.array): array with the predictions
        ------------------------------------------------------
        Returns:
            threshold (float): threshold for the model
            fpr@5FPR (float): false positive rate of the model
            recall@5FPR (float): recall of the model
            tnr@5FPR (float): true negative rate of the model
            accuracy@5FPR (float): accuracy of the model
            precision@5FPR (float): precision of the model
            f1_score@5FPR (float): f1 score of the model
        """
        metrics = evaluate_business_constraint(y_test, predictions)
        return metrics
    
    def evaluate_fairness(self, x_test, y_test, predictions, sensitive_attribute, attribute_threshold):
        """Evaluate the model using the Aequitas library.
        ------------------------------------------------------
        Args:
            x_test (pd.DataFrame): dataframe with the test features
            y_test (pd.Series): series with the test labels
            predictions (np.array): array with the predictions
            sensitive_attribute (str): name of the sensitive attribute
            attribute_threshold (float): threshold for the sensitive attribute
        ------------------------------------------------------
        Returns:
            fpr_ratio (float): false positive rate ratio of the model
            fnr_ratio (float): false negative rate ratio of the model
        """
        metrics = evaluate_fairness(x_test, y_test, predictions, sensitive_attribute, attribute_threshold)
        return metrics
    
