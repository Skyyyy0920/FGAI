from utils import *
import scipy.sparse as sp


class StandardTrainer:
    def __init__(self, standard_model, criterion, optimizer, args):
        self.model = standard_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = args.device
        self.num_epochs = args.num_epochs

    def train(self, features, adj, label, train_idx, valid_idx):
        original_outputs, original_graph_repr, original_att = None, None, None
        for epoch in range(self.num_epochs):
            self.model.train()

            original_outputs, original_graph_repr, original_att = self.model(features, adj)
            loss = self.criterion(original_outputs[train_idx], label[train_idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs, _, _ = self.model(features, adj)
                val_loss = self.criterion(val_outputs[valid_idx], label[valid_idx])
                val_pred = torch.argmax(val_outputs[valid_idx], dim=1)
                val_accuracy = accuracy_score(label[valid_idx].cpu(), val_pred.cpu())

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')

        return original_outputs, original_graph_repr, original_att


class FGAITrainer:
    def __init__(self, FGAI, optimizer, attacker_delta, attacker_rho, args):
        self.model = FGAI
        self.optimizer = optimizer
        self.attacker_delta = attacker_delta
        self.attacker_rho = attacker_rho
        self.device = args.device
        self.K = args.K
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.num_epochs = args.num_epochs

    def train(self, features, adj, label, idx_split, orig_outputs, orig_graph_repr, orig_att):
        train_idx, valid_idx, test_idx = idx_split
        for epoch in range(self.num_epochs):
            self.model.train()

            FGAI_outputs, FGAI_graph_repr, FGAI_att = self.model(features, adj)

            # 1. Closeness of Prediction
            closeness_of_prediction_loss = TVD(FGAI_outputs, orig_outputs)

            # 2. Constraint of Stability. Perturb δ(x) to ensure robustness of FGAI
            # TODO: train_idx????????
            adj_delta, feats_delta = self.attacker_delta.attack(self.model, adj, features, test_idx, None)
            new_outputs, new_graph_repr, new_att = self.model(torch.cat((features, feats_delta), dim=0), adj_delta)
            adversarial_loss = TVD(new_outputs[:features.shape[0]], FGAI_outputs)

            # 3. Stability of Explanation. Perturb 𝝆(x) to ensure robustness of explanation of FGAI
            adj_rho, feats_rho = self.attacker_rho.attack(self.model, adj, features, test_idx, None)
            new_outputs_2, new_graph_repr_2, new_att_2 = self.model(torch.cat((features, feats_rho), dim=0), adj_rho)
            stability_of_explanation_loss = 0
            for i in range(orig_att.shape[1]):
                stability_of_explanation_loss += topK_overlap_loss(new_att_2[:, i][:features.shape[0]], FGAI_att[:, i],
                                                                   adj, self.K, 'l1')

            # 4. Similarity of Explanation
            similarity_of_explanation_loss = 0
            for i in range(orig_att.shape[1]):
                similarity_of_explanation_loss += topK_overlap_loss(FGAI_att[:, i], orig_att[:, i], adj, self.K, 'l1')

            loss = closeness_of_prediction_loss + adversarial_loss * self.lambda_1 + \
                   stability_of_explanation_loss * self.lambda_2 + similarity_of_explanation_loss * self.lambda_3

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs, orig_graph_repr, _ = self.model(g, features)
                val_pred = torch.argmax(val_outputs[valid_idx], dim=1)
                val_accuracy = accuracy_score(label[valid_idx].cpu(), val_pred.cpu())

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val Accuracy: {val_accuracy:.4f}')
