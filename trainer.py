import torch
import logging
from utils import *


class StandardTrainer:
    def __init__(self, standard_model, criterion, optimizer, args):
        self.model = standard_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = args.device
        self.num_epochs = args.num_epochs

    def train(self, g, features, m_l):
        train_mask, train_label, val_mask, val_label = m_l
        original_outputs, original_graph_repr, original_att = None, None, None
        for epoch in range(self.num_epochs):
            self.model.train()
            original_outputs, original_graph_repr, original_att = self.model(g, features)
            loss = self.criterion(original_outputs[train_mask], train_label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs, _, _ = self.model(g, features)
                val_loss = self.criterion(val_outputs[val_mask], val_label)
                val_pred = torch.argmax(val_outputs[val_mask], dim=1)
                val_accuracy = accuracy_score(val_label.cpu(), val_pred.cpu())

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')

        # Fix the parameters of the standard model
        for pp in self.model.parameters():
            pp.requires_grad = False

        return original_outputs, original_graph_repr, original_att


class FGAITrainer:
    def __init__(self, FGAI, criterion, optimizer, PGDer, args):
        self.model = FGAI
        self.criterion = criterion
        self.optimizer = optimizer
        self.PGDer = PGDer
        self.device = args.device
        self.K = args.K
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_4 = args.lambda_4
        self.num_epochs = args.num_epochs

    def train(self, g, features, m_l, orig_outputs, orig_graph_repr, orig_att):
        train_mask, train_label, val_mask, val_label = m_l
        for epoch in range(self.num_epochs):
            self.model.train()

            FGAI_outputs, FGAI_graph_repr, FGAI_att = self.model(g, features)

            # 1. Closeness of Prediction
            closeness_of_prediction_loss = TVD(FGAI_outputs[train_mask], orig_outputs[train_mask])

            # 2. Constraint of Stability. Perturb δ(x) to ensure robustness of FGAI
            feats_delta = self.PGDer.perturb_delta(features, train_mask, FGAI_outputs, g, self.model)
            new_outputs, new_graph_repr, new_att = self.model(g, feats_delta)
            adversarial_loss = TVD(new_outputs[train_mask], FGAI_outputs[train_mask])

            # 3. Stability of Explanation. Perturb 𝝆(x) to ensure robustness of explanation of FGAI
            feats_rho = self.PGDer.perturb_rho(features, FGAI_att, g, self.model)
            new_outputs_2, new_graph_repr_2, new_att_2 = self.model(g, feats_rho)
            stability_of_explanation_loss = 0
            for i in range(orig_att.shape[1]):
                stability_of_explanation_loss += topK_overlap_loss(new_att_2[:, i], FGAI_att[:, i], g, self.K, 'l1')

            # 4. Similarity of Explanation
            similarity_of_explanation_loss = 0
            for i in range(orig_att.shape[1]):
                similarity_of_explanation_loss += topK_overlap_loss(FGAI_att[:, i], orig_att[:, i], g, self.K, 'l1')

            loss = closeness_of_prediction_loss + adversarial_loss * self.lambda_1 + \
                   stability_of_explanation_loss * self.lambda_2 + similarity_of_explanation_loss * self.lambda_3

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs, orig_graph_repr, _ = self.model(g, features)
                val_loss = self.criterion(val_outputs[val_mask], val_label)
                val_pred = torch.argmax(val_outputs[val_mask], dim=1)
                val_accuracy = accuracy_score(val_label.cpu(), val_pred.cpu())

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')
