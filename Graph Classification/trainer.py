from utils import *
import torch.nn as nn
import torch.nn.functional as F


class VanillaTrainer:
    def __init__(self, standard_model, criterion, optimizer, args):
        self.model = standard_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = args.device
        self.num_epochs = args.num_epochs

    def train(self, train_loader, valid_loader):
        original_outputs, original_graph_repr, original_att = None, None, None
        for epoch in range(self.num_epochs):
            self.model.train()
            loss_list = []
            for batched_graph, labels in train_loader:
                labels = labels.squeeze().to(self.device)
                feats = batched_graph.ndata['feat'].to(self.device)

                original_outputs, original_att = self.model(feats.to(self.device), batched_graph.to(self.device))
                loss = self.criterion(original_outputs, labels.long())
                loss_list.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            logging.info(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {np.mean(loss_list):.4f}")

            self.model.eval()
            with torch.no_grad():
                loss_list = []
                pred_list, label_list = [], []
                for batched_graph, labels in valid_loader:
                    labels = labels.view(-1).to(self.device)
                    feats = batched_graph.ndata['feat'].float().to(self.device)

                    logits, _ = self.model(feats, batched_graph.to(self.device))

                    loss = self.criterion(logits, labels.long())
                    loss_list.append(loss.item())

                    predicted = logits.argmax(dim=1)
                    pred_list = pred_list + predicted.tolist()
                    label_list = label_list + labels.tolist()

                accuracy = accuracy_score(label_list, pred_list)

            logging.info(f'Val Loss: {np.mean(loss_list):.4f} | Accuracy: {accuracy:.4f}')

        return original_outputs, original_graph_repr, original_att


class FGAIGraphTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 attacker,
                 args,
                 loss_func=nn.CrossEntropyLoss()):
        self.model = model.to(args.device)
        self.optimizer = optimizer
        self.attacker = attacker
        self.args = args
        self.loss_func = loss_func
        self.args.lambda_adv = 0.2
        self.args.lambda_att = 0.2

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batched_graph, labels in train_loader:
            labels = labels.to(self.args.device)
            feats = batched_graph.ndata['feat'].to(self.args.device)

            adv_graphs, adv_feats, _, _ = self.attacker.perturb_batch(batched_graph, feats)

            orig_logits, orig_att = self.model(feats, batched_graph.to(feats.device))
            adv_logits, adv_att = self.model(adv_feats, adv_graphs.to(feats.device))

            cls_loss = self.loss_func(orig_logits, labels.long())
            adv_loss = F.kl_div(
                F.log_softmax(adv_logits, dim=1),
                F.softmax(orig_logits.detach(), dim=1)
            )
            att_loss = self._attention_similarity(orig_att, adv_att)

            loss = cls_loss + self.args.lambda_adv * adv_loss + self.args.lambda_att * att_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _attention_similarity(self, att1, att2):
        return 1 - F.cosine_similarity(
            att1.flatten(1),
            att2.flatten(1)
        ).mean()

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        all_orig, all_adv = [], []

        with torch.no_grad():
            for batched_graph, labels in loader:
                feats = batched_graph.ndata['feat'].to(self.args.device)
                orig_logits, orig_att = self.model(feats, batched_graph.to(self.args.device))

                adv_graphs, adv_feats, _, _ = self.attacker.perturb_batch(batched_graph, feats)
                adv_logits, adv_att = self.model(adv_feats, adv_graphs.to(self.args.device))

                all_orig.append(orig_logits.softmax(dim=1))
                all_adv.append(adv_logits.softmax(dim=1))
                all_preds.extend(adv_logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())

        orig_probs = torch.cat(all_orig)
        adv_probs = torch.cat(all_adv)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'tvd': 0.5 * (orig_probs - adv_probs).abs().sum(dim=1).mean().item(),
        }
        return metrics
