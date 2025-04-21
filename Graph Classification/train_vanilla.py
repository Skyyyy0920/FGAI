import time
import yaml
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *
from models import GATGraphClassifier
from load_dataset import load_dataset
from trainer import VanillaTrainer
from attackers import GraphRandomAttacker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # dataset = 'ogbg-molhiv'
    dataset = 'DD'
    # dataset = 'COLLAB'
    # dataset = 'ENZYMES'
    # dataset = 'MUTAG'
    # dataset = 'ogbg-ppa'

    # base_model = 'GAT'
    base_model = 'GATv2'
    # base_model = 'GT'

    with open(f"./optimized_hyperparameter_configurations/{base_model}/{dataset}.yml", 'r') as file:
        args = yaml.full_load(file)
    args = argparse.Namespace(**args)
    args.device = device
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("checkpoints", f"{base_model}+vanilla", f"{dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    logging.info(f"base model: {base_model}")

    train_loader, valid_loader, test_loader, in_feats, num_classes = load_dataset(args)

    if base_model == 'GAT':
        model = GATGraphClassifier(
            feat_size=in_feats,
            hidden_size=args.hid_dim,
            n_classes=num_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            residual=True,
            readout='attention'
        ).to(args.device)

    elif base_model == 'GATv2':
        model = GATGraphClassifier(
            feat_size=in_feats,
            hidden_size=args.hid_dim,
            n_classes=num_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            residual=True,
            readout='attention',
            version='v2'
        ).to(args.device)

    elif base_model == 'GT':
        model = GTGraphClassifier(
            feats_size=features.shape[1],
            hidden_size=args.hid_dim,
            out_size=num_classes,
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

    trainer = VanillaTrainer(model, criterion, optimizer, args)
    trainer.train(train_loader, test_loader)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))

    attacker = GraphRandomAttacker(
        edge_add_ratio=0.2,
        inject_node_max=2,
        attack_mode='mixed'
    )
    pred_list, label_list = [], []
    all_orig_outputs, all_new_outputs = [], []
    all_orig_att, all_new_att = [], []
    for batched_graph, labels in tqdm(test_loader):
        labels = labels.to(device)
        feats = batched_graph.ndata['feat'].to(device)
        adv_graphs, adv_feats, node_mask, edge_mask = attacker.perturb_batch(batched_graph, feats)

        with torch.no_grad():
            orig_logits, orig_att = model(feats, batched_graph.to(device))
            adv_feats = adv_feats[:adv_graphs.num_nodes()]
            logits, att = model(adv_feats, adv_graphs.to(device))

        predicted = logits.argmax(dim=1)
        pred_list = pred_list + predicted.tolist()
        label_list = label_list + labels.tolist()

        all_orig_outputs.append(orig_logits)
        all_new_outputs.append(logits)
        orig_att = orig_att.mean(dim=1, keepdim=True)  # 多头注意力平均 [N,1,1]
        new_att = att.mean(dim=1, keepdim=True)
        all_orig_att.append(orig_att)
        all_new_att.append(new_att[edge_mask])

    accuracy = accuracy_score(label_list, pred_list)
    logging.info(f'Test Accuracy after perturbed: {accuracy:.4f}')

    orig_outputs = torch.cat(all_orig_outputs)
    new_outputs = torch.cat(all_new_outputs)
    orig_att = torch.cat(all_orig_att)  # [total_nodes,1,1]
    new_att = torch.cat(all_new_att)[:orig_att.shape[0]]
    TVD_score = TVD(orig_outputs, new_outputs) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")
