import yaml
import argparse

from models import *
from trainer import *
from attackers import *

from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
import pandas as pd
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    # ==================================================================================================
    # 1. Choose the dataset, base model
    # ==================================================================================================
    # dataset = 'cora'
    # dataset = 'pubmed'
    dataset = 'citeseer'

    base_model = 'GAT'
    # base_model = 'GATv2'
    # base_model = 'GT'

    # ==================================================================================================
    # 2. Get experiment args and seed
    # ==================================================================================================
    current_dir = os.getcwd()
    print("Current work dir：", current_dir)
    new_dir = current_dir + "/Link Prediction"
    os.chdir(new_dir)
    with open(f"./hyperparameter_configurations/{base_model}/{dataset}.yml", 'r') as file:
        args = yaml.full_load(file)
    args = argparse.Namespace(**args)
    args.device = device
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("checkpoints", f"{base_model}+vanilla", f"{dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    logging.info(f"base model: {base_model}")

    # ==================================================================================================
    # 3. Prepare data
    # ==================================================================================================
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    dataset = Planetoid('./dataset', name='Cora', transform=transform)
    train_data, val_data, test_data = dataset[0]
    pos_enc_size = 8
    out_size = 64

    # ==================================================================================================
    # 4. Build models, define overall loss and optimizer
    # ==================================================================================================
    if base_model == 'GAT':
        model = GATLinkPredictor(
            feats_size=dataset.num_features,
            hidden_size=args.hid_dim,
            out_size=out_size,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            layer_norm=False
        ).to(device)

    elif base_model == 'GATv2':
        model = GATv2NodeClassifier(
            feats_size=dataset.num_features,
            hidden_size=args.hid_dim,
            out_size=out_size,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            layer_norm=False
        ).to(device)

    elif base_model == 'GT':
        model = GTNodeClassifier(
            feats_size=dataset.num_features,
            hidden_size=args.hid_dim,
            out_size=out_size,
            pos_enc_size=pos_enc_size,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            layer_norm=False
        ).to(device)
        pos_enc_path = f"./GT_pos_encoding/{dataset}_pos_enc.pth"
        if os.path.exists(pos_enc_path):
            pos_enc = torch.load(pos_enc_path)
        else:
            in_degrees = torch.tensor(adj.sum(axis=0)).squeeze()
            pos_enc = laplacian_pe(adj, in_degrees, k=pos_enc_size, padding=True).to(device)
            torch.save(pos_enc, pos_enc_path)
        model.pos_enc = pos_enc
        pos_enc_per_path = f"./GT_pos_encoding/{dataset}_pos_enc_perturbed.pth"
        if os.path.exists(pos_enc_per_path):
            model.pos_enc_ = torch.load(pos_enc_per_path)
            need_update = False
        else:
            need_update = True

    else:
        raise ValueError(f"Unknown base model name: {base_model}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {model}")
    logging.info(f"Optimizer: {optimizer}")


    # ==================================================================================================
    # 5. Training
    # ==================================================================================================
    def train():
        model.train()
        optimizer.zero_grad()
        z, _, att = model.encode(train_data.x, train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss


    @torch.no_grad()
    def test(data):
        model.eval()
        z, _, att = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        # precision_k = precision_at_k(data.edge_label.cpu().numpy(), out.cpu().numpy(), k=10)
        # print("@10:", precision_k)
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


    best_val_auc = final_test_auc = 0
    # for epoch in range(1, 81):
    for epoch in range(args.num_epochs):
        loss = train()
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')

    # z = model.encode(test_data.x, test_data.edge_index)
    # final_edge_index = model.decode_all(z)

    model.eval()
    z, _, orig_att = model.encode(test_data.x, test_data.edge_index)
    orig_outputs = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))

    attacker = PGD(
        epsilon=args.epsilon,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-1,
        feat_lim_max=1,
        eval_metric=roc_auc_score,
        device=device
    )
    test_nodes = torch.unique(test_data.edge_index.flatten())
    adj_delta, feats_delta = attacker.attack(model, test_data, train_data.x, test_nodes)
    sp.save_npz(os.path.join(save_dir, 'adj_delta.npz'), adj_delta)
    torch.save(feats_delta, os.path.join(save_dir, 'feats_delta.pth'))

    if base_model == 'GT' and need_update:
        in_degrees = torch.tensor(adj_delta.sum(axis=0)).squeeze()
        pos_enc_ = laplacian_pe(adj_delta, in_degrees, k=pos_enc_size, padding=True).to(device)
        torch.save(pos_enc_, f'./{dataset}_pos_enc_perturbed.pth')
        model.pos_enc_ = pos_enc_

    feats_ = torch.cat((test_data.x, feats_delta), dim=0)
    row_indices, col_indices = adj_delta.nonzero()
    edges_delta = torch.tensor([row_indices, col_indices])
    z, _, new_att = model.encode(feats_, edges_delta)
    print(len(new_att))
    print(len(orig_att))
    new_outputs = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
    new_outputs, new_att = new_outputs[:orig_outputs.shape[0]], new_att[:orig_att.shape[0]]
    accuracy = roc_auc_score(test_data.edge_label.detach().cpu().numpy(), new_outputs.detach().cpu().numpy())
    logging.info(f"Accuracy after Injection Attack: {accuracy:.4f}")

    TVD_score = TVD(orig_outputs, new_outputs) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")
