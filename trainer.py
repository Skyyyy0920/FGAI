import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from utils import *
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, model, criterion, optimizer, PGDer, X_PGDer, args):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.PGDer = PGDer
        self.X_PGDer = X_PGDer
        self.device = args.device
        self.K = args.K
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_4 = args.lambda_4
        self.num_epochs = args.num_epochs

    def train(self, g, features, train_mask, train_label, val_mask, val_label):
        for epoch in range(self.num_epochs):
            self.model.train()
            outputs, graph_repr, original_att = self.model(g, features)

            # 1. Closeness of Prediction
            closeness_of_prediction_loss = self.criterion(outputs[train_mask], train_label)

            # Perturb e(x) to ensure robustness of models
            new_features = self.PGDer.perturb(batch_tvd, features, train_mask, train_label, g, self.model)
            new_outputs, new_graph_repr, new_att = self.model(g, new_features)

            # 2. Stability
            adversarial_loss = self.criterion(new_outputs[train_mask], train_label)

            # 3. Similarity of explanation
            similarity_of_explanation_loss = topk_overlap_loss(new_att, original_att, self.K, 'l1')

            # loss = closeness_of_prediction_loss + \
            #        adversarial_loss * self.lambda_1 + \
            #        similarity_of_explanation_loss * self.lambda_2
            loss = closeness_of_prediction_loss + adversarial_loss * self.lambda_1

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs, graph_repr, _ = self.model(g, features)
                val_loss = self.criterion(val_outputs[val_mask], val_label)
                val_preds = torch.argmax(val_outputs[val_mask], dim=1)
                val_accuracy = accuracy_score(val_label.cpu(), val_preds.cpu())

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')

    def evaluate(self, g, features, test_mask, test_label):
        self.model.eval()
        with torch.no_grad():
            test_outputs, graph_rep, _ = self.model(g, features)
            test_loss = self.criterion(test_outputs[test_mask], test_label)
            test_preds = torch.argmax(test_outputs[test_mask], dim=1)
            test_accuracy = accuracy_score(test_label.cpu(), test_preds.cpu())

        logging.info(f'Test Loss: {test_loss.item():.4f} | Test Accuracy: {test_accuracy:.4f}')
