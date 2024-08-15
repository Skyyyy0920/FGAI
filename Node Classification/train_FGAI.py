import yaml
import argparse
import pandas as pd

from models import *
from trainer import *
from attackers import *
from load_dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # ==================================================================================================
    # 1. Choose the dataset, base model
    # ==================================================================================================
    dataset = 'amazon_photo'
    # dataset = 'amazon_cs'
    # dataset = 'coauthor_cs'
    # dataset = 'coauthor_phy'
    # dataset = 'pubmed'
    # dataset = 'ogbn-arxiv'

    base_model = 'GAT'
    # base_model = 'GATv2'
    # base_model = 'GT'

    # ==================================================================================================
    # 2. Get experiment args and seed
    # ==================================================================================================
    current_dir = os.getcwd()
    print("Current work dir：", current_dir)
    new_dir = current_dir + "/Node Classification"
    os.chdir(new_dir)
    with open(f"./hyperparameter_configurations/{base_model}/FGAI_{dataset}.yml", 'r') as file:
        args = yaml.full_load(file)
    args = argparse.Namespace(**args)
    args.device = device
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("checkpoints", f"{base_model}+FGAI", f"{dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    logging.info(f"base model: {base_model}")

    # ==================================================================================================
    # 3. Prepare data
    # ==================================================================================================
    g, adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    idx_split = train_idx, valid_idx, test_idx
    pos_enc_size = 8

    # ==================================================================================================
    # 4. Build models, define overall loss and optimizer
    # ==================================================================================================
    if base_model == 'GAT':
        model = GATNodeClassifier(
            feats_size=features.shape[1],
            hidden_size=args.hid_dim,
            out_size=num_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            layer_norm=False
        ).to(device)

    elif base_model == 'GATv2':
        model = GATNodeClassifier(
            feats_size=features.shape[1],
            hidden_size=args.hid_dim,
            out_size=num_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            v2=True,
            layer_norm=False
        ).to(device)

    elif base_model == 'GT':
        model = GTNodeClassifier(
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

    attacker_delta = PGD(
        epsilon=args.epsilon,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-1,
        feat_lim_max=1,
        loss=TVD,
        device=device
    )
    attacker_rho = PGD(
        epsilon=args.epsilon,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-1,
        feat_lim_max=1,
        loss=topK_overlap_loss,
        K=args.K_rho,
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {model}")
    logging.info(f"Optimizer: {optimizer}")

    # ==================================================================================================
    # 5. Load pre-trained vanilla model
    # ==================================================================================================
    tim = '00-40'
    model.load_state_dict(torch.load(f"./checkpoints/{base_model}+vanilla/{dataset}_{tim}/model_parameters.pth"))
    vanilla_outputs, _, vanilla_att = evaluate_node_level(model, features, adj, label, test_idx)

    # ==================================================================================================
    # 6. Training
    # ==================================================================================================
    trainer = FGAITrainer(model, optimizer, attacker_delta, attacker_rho, args)
    trainer.train(features, adj, label, idx_split, vanilla_outputs, vanilla_att, save_dir)

    orig_outputs, _, orig_att = evaluate_node_level(model, features, adj, label, test_idx)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))

    attacker = PGD(
        epsilon=args.epsilon,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-1,
        feat_lim_max=1,
        device=device
    )
    adj_delta, feats_delta = attacker.attack(model, adj, features, test_idx, None)
    sp.save_npz(os.path.join(save_dir, 'adj_delta.npz'), adj_delta)
    torch.save(feats_delta, os.path.join(save_dir, 'feats_delta.pth'))

    if base_model == 'GT' and need_update:
        in_degrees = torch.tensor(adj_delta.sum(axis=0)).squeeze()
        pos_enc_ = laplacian_pe(adj_delta, in_degrees, k=pos_enc_size, padding=True).to(device)
        torch.save(pos_enc_, f'./{dataset}_pos_enc_perturbed.pth')
        model.pos_enc_ = pos_enc_

    feats_ = torch.cat((features, feats_delta), dim=0)
    new_outputs, _, new_att = model(feats_, adj_delta)
    new_outputs, new_att = new_outputs[:orig_outputs.shape[0]], new_att[:orig_att.shape[0]]
    pred = torch.argmax(new_outputs[test_idx], dim=1)
    accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
    logging.info(f"Accuracy after Injection Attack: {accuracy:.4f}")

    TVD_score = TVD(orig_outputs, new_outputs) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    f_pos_list, f_neg_list = compute_fidelity(model, adj, features, label, test_idx, orig_att)
    logging.info(f"fidelity_pos: {f_pos_list}")
    logging.info(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, f'{base_model}+FGAI_F.txt'), sep=',', index=False)

    f_pos_list, f_neg_list = compute_fidelity_attacked(model, adj, features, adj_delta, feats_, label, test_idx,
                                                       new_att)
    logging.info(f"fidelity_pos_after_attack: {f_pos_list}")
    logging.info(f"fidelity_neg_after_attack: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, f'{base_model}+FGAI_F_after_attack.txt'), sep=',', index=False)

    attacker_ = PGD(
        epsilon=0.00001,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-0.001,
        feat_lim_max=0.001,
        device=device,
        mode='Modification Attack'
    )
    adj_delta_, feats_delta_ = attacker_.attack(model, adj, features, test_idx, None)
    sp.save_npz(os.path.join(save_dir, 'adj_delta_.npz'), adj_delta_)
    torch.save(feats_delta_, os.path.join(save_dir, 'feats_delta_.pth'))

    new_outputs_, new_graph_repr, new_att_ = model(feats_delta_, adj_delta_)
    pred = torch.argmax(new_outputs_[test_idx], dim=1)
    accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
    logging.info(f"Accuracy after Modification Attack: {accuracy:.4f}")

    TVD_score = TVD(orig_outputs, new_outputs_) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att_) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    f_pos_list, f_neg_list = compute_fidelity_attacked(model, adj, features, adj_delta_, feats_delta_,
                                                       label, test_idx, new_att)
    logging.info(f"fidelity_pos_after_attack: {f_pos_list}")
    logging.info(f"fidelity_neg_after_attack: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, f'{base_model}+FGAI_F_after_attack_.txt'), sep=',', index=False)




