from utils import *
import matplotlib
import torch.nn.functional as F
import matplotlib.pyplot as plt

matplotlib.use('Agg')


class VanillaTrainer(object):
    def __init__(self, standard_model, criterion, optimizer, args):
        self.model = standard_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = args.device
        self.num_epochs = args.num_epochs

    def train(self, features, adj, label, idx_split):
        train_idx, valid_idx, test_idx = idx_split
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
                # roc_auc = roc_auc_score(label[valid_idx].cpu(), val_pred.cpu())
                # print(roc_auc)

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')


class AdvTrainer(object):
    def __init__(self, model, optimizer, criterion, attacker, args):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.attacker = attacker
        self.device = args.device
        self.num_epochs = args.num_epochs

    def train(self, feats, adj, label, idx_split):
        train_idx, valid_idx, test_idx = idx_split
        for epoch in range(self.num_epochs):
            self.model.train()

            outputs, graph_repr, att = self.model(feats, adj)
            loss = self.criterion(outputs[train_idx], label[train_idx])

            target_mask = torch.ones(feats.shape[0]).bool()
            adj_delta, feats_delta = self.attacker.attack(self.model, adj, feats, target_mask, None)
            outputs, graph_repr, att = self.model(torch.cat((feats, feats_delta), dim=0), adj_delta)
            outputs, graph_repr, att = outputs[:feats.shape[0]], graph_repr[:feats.shape[0]], att[:feats.shape[0]]
            loss += self.criterion(outputs[train_idx], label[train_idx]) * 0.1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs, orig_graph_repr, _ = self.model(feats, adj)
                val_pred = torch.argmax(val_outputs[valid_idx], dim=1)
                val_accuracy = accuracy_score(label[valid_idx].cpu(), val_pred.cpu())
                val_loss = F.cross_entropy(val_outputs[valid_idx], label[valid_idx])

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val loss: {val_loss} | Val Accuracy: {val_accuracy:.4f}')


class FGAITrainer(object):
    def __init__(self, FGAI, optimizer, attacker_delta, attacker_rho, args, loss_type=topK_overlap_loss):
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
        self.early_stopping = args.early_stopping
        self.overlap_loss = loss_type

    def train(self, train_data, valid_data, test_data, orig_outputs, orig_att, save_dir):
        features = train_data.x
        edges = test_data.edge_index
        row_indices = edges[0].cpu().numpy()
        col_indices = edges[1].cpu().numpy()
        data = np.ones(len(row_indices))
        num_nodes = len(features)
        adj = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

        best_val_loss = float('inf')
        current_patience = 0
        early_stopping_flag = False
        closeness_losses = []
        adversarial_losses = []
        stability_losses = []
        similarity_losses = []
        total_losses = []
        target_mask = torch.unique(train_data.edge_index.flatten())
        for epoch in range(self.num_epochs):
            self.model.train()

            z, FGAI_graph_repr, FGAI_att = self.model.encode(features, train_data.edge_index)
            FGAI_outputs = self.model.decode(z, train_data.edge_label_index).view(-1).sigmoid()

            # 1. Closeness of Prediction
            closeness_of_prediction_loss = TVD(FGAI_outputs, orig_outputs)

            # 2. Similarity of Explanation
            similarity_of_explanation_loss = topK_overlap_loss(FGAI_att, orig_att[:FGAI_att.shape[0]], adj, self.K,
                                                               'l1', 'graph')

            # 3. Constraint of Stability. Perturb δ(x) to ensure robustness of FGAI
            adj_delta, feats_delta = self.attacker_delta.attack(self.model, train_data, features, target_mask)
            row_indices, col_indices = adj_delta.nonzero()
            edges_delta = torch.tensor([row_indices, col_indices])
            z, new_graph_repr, new_att = self.model.encode(torch.cat((features, feats_delta), dim=0),
                                                           edges_delta)
            new_outputs = self.model.decode(z, train_data.edge_label_index).view(-1).sigmoid()
            adversarial_loss = TVD(new_outputs[:FGAI_outputs.shape[0]], orig_outputs)

            # 4. Stability of Explanation. Perturb 𝝆(x) to ensure robustness of explanation of FGAI
            adj_rho, feats_rho = self.attacker_rho.attack(self.model, train_data, features, target_mask)
            row_indices, col_indices = adj_rho.nonzero()
            edges_rho = torch.tensor([row_indices, col_indices])
            z, new_graph_repr_2, new_att_2 = self.model.encode(torch.cat((features, feats_rho), dim=0), edges_rho)
            new_outputs_2 = self.model.decode(z, train_data.edge_label_index).view(-1).sigmoid()
            stability_of_explanation_loss = topK_overlap_loss(new_att_2[:FGAI_att.shape[0]], FGAI_att, adj, self.K,
                                                              'l1', 'graph')

            loss = closeness_of_prediction_loss + adversarial_loss * self.lambda_1 + \
                   stability_of_explanation_loss * self.lambda_2 + similarity_of_explanation_loss * self.lambda_3

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                z, _, _ = self.model.encode(features, train_data.edge_index)
                val_outputs = self.model.decode(z, valid_data.edge_label_index).view(-1).sigmoid()
                val_loss = F.cross_entropy(val_outputs, valid_data.edge_label)
                val_roc_auc = roc_auc_score(valid_data.edge_label.cpu().numpy(), val_outputs.cpu().numpy())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= self.early_stopping:
                    logging.info(f"Early stopping at epoch {epoch + 1}...")
                    early_stopping_flag = True

            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}] | Train Loss: {loss.item():.4f} | '
                         f'Val loss: {val_loss.item()} | Val ROC_AUC: {val_roc_auc:.4f}')
            logging.info(f'Loss item: {closeness_of_prediction_loss.item()}, '
                         f'{adversarial_loss.item() * self.lambda_1}, '
                         f'{stability_of_explanation_loss.item() * self.lambda_2}, '
                         f'{similarity_of_explanation_loss.item() * self.lambda_3}')
            total_losses.append(loss.item())
            closeness_losses.append(closeness_of_prediction_loss.item())
            adversarial_losses.append(adversarial_loss.item() * self.lambda_1)
            stability_losses.append(stability_of_explanation_loss.item() * self.lambda_2)
            similarity_losses.append(similarity_of_explanation_loss.item() * self.lambda_3)

            if early_stopping_flag:
                break

        plt.figure(figsize=(10, 6))
        epochs = range(0, epoch)

        plt.plot(epochs, closeness_losses[1:], label='Closeness Loss')
        plt.plot(epochs, adversarial_losses[1:], label='Adversarial Loss')
        plt.plot(epochs, stability_losses[1:], label='Stability of Explanation Loss')
        plt.plot(epochs, similarity_losses[1:], label='Similarity of Explanation Loss')
        plt.plot(epochs, total_losses[1:], label='Total Loss', linestyle='--', color='black')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Components Over Epochs')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/loss_components.pdf')
