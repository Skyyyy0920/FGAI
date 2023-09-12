import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from utils import *
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, standard_model, FGAI, criterion, optimizer, PGDer, PGDer_2, args):
        self.standard_model = standard_model
        self.FGAI = FGAI
        self.criterion = criterion
        self.optimizer = optimizer
        self.PGDer = PGDer
        self.PGDer_2 = PGDer_2
        self.device = args.device
        self.K = args.K
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_4 = args.lambda_4
        self.num_epochs = args.num_epochs

    def train_standard(self, g, features, m_l):
        train_mask, train_label, val_mask, val_label = m_l
        original_outputs, original_graph_repr, original_att = None, None, None
        for epoch in range(self.num_epochs):
            self.standard_model.train()
            original_outputs, original_graph_repr, original_att = self.standard_model(g, features)
            loss = self.criterion(original_outputs[train_mask], train_label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.standard_model.eval()
            with torch.no_grad():
                val_outputs, _, _ = self.standard_model(g, features)
                val_loss = self.criterion(val_outputs[val_mask], val_label)
                val_pred = torch.argmax(val_outputs[val_mask], dim=1)
                val_accuracy = accuracy_score(val_label.cpu(), val_pred.cpu())

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')
        return original_outputs, original_graph_repr, original_att

    def train_FGAI(self, g, features, m_l, original_outputs, original_graph_repr, original_att):
        train_mask, train_label, val_mask, val_label = m_l
        for epoch in range(self.num_epochs):
            self.FGAI.train()

            # 1. Closeness of Prediction
            closeness_of_prediction_loss = self.criterion(original_outputs[train_mask], train_label)

            # 2. Constraint of Stability. Perturb e(x) to ensure robustness of models
            new_features = self.PGDer.perturb(batch_TVD, features, train_mask, train_label, g, self.standard_model)
            new_outputs, new_graph_repr, new_att = self.standard_model(g, new_features)
            adversarial_loss = self.criterion(new_outputs[train_mask], train_label)

            # 3. Stability of Explanation. Perturb 𝝆(x) to ensure robustness of explanation of FGAI
            new_features_2 = self.PGDer_2.perturb(batch_TVD, features, train_mask, train_label, g, self.standard_model)
            new_outputs_2, new_graph_repr_2, new_att_2 = self.standard_model(g, new_features_2)
            stability_of_explanation_loss = 0
            for i in range(original_att.shape[1]):
                stability_of_explanation_loss += topk_overlap_loss(new_att_2[:, i], new_att[:, i], g, self.K, 'l1')

            # 4. Similarity of Explanation
            similarity_of_explanation_loss = 0
            for i in range(original_att.shape[1]):
                similarity_of_explanation_loss += topk_overlap_loss(new_att[:, i], original_att[:, i], g, self.K, 'l1')

            loss = closeness_of_prediction_loss + adversarial_loss * self.lambda_1 + \
                   stability_of_explanation_loss * self.lambda_2 + similarity_of_explanation_loss * self.lambda_3

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.standard_model.eval()
            with torch.no_grad():
                val_outputs, original_graph_repr, _ = self.standard_model(g, features)
                val_loss = self.criterion(val_outputs[val_mask], val_label)
                val_pred = torch.argmax(val_outputs[val_mask], dim=1)
                val_accuracy = accuracy_score(val_label.cpu(), val_pred.cpu())

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')

    def evaluate(self, g, features, test_mask, test_label, model='FGAI'):
        if model == 'standard':
            self.standard_model.eval()
            with torch.no_grad():
                test_outputs, graph_rep, _ = self.standard_model(g, features)
                test_loss = self.criterion(test_outputs[test_mask], test_label)
                test_pred = torch.argmax(test_outputs[test_mask], dim=1)
                test_accuracy = accuracy_score(test_label.cpu(), test_pred.cpu())
        elif model == 'FGAI':
            self.FGAI.eval()
            with torch.no_grad():
                test_outputs, graph_rep, _ = self.FGAI(g, features)
                test_loss = self.criterion(test_outputs[test_mask], test_label)
                test_pred = torch.argmax(test_outputs[test_mask], dim=1)
                test_accuracy = accuracy_score(test_label.cpu(), test_pred.cpu())

        logging.info(f'Test Loss: {test_loss.item():.4f} | Test Accuracy: {test_accuracy:.4f}')
