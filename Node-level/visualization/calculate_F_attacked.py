import os
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sp
from models import GATNodeClassifier, GATv2NodeClassifier, GTNodeClassifier
from load_dataset import load_dataset
from scipy.sparse import csr_matrix
from utils import evaluate_node_level

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

if __name__ == '__main__':
    # dataset = 'amazon_photo'
    # dataset = 'amazon_cs'
    # dataset = 'coauthor_cs'
    dataset = 'coauthor_phy'
    # dataset ='ogbn-arxiv'

    with open(f"./GAT/optimized_hyperparameter_configurations/FGAI_{dataset}.yml", 'r') as file:
        args = yaml.safe_load(file)
    args = argparse.Namespace(**args)
    args.device = device

    # ==================================================================================================
    # 4. Prepare data
    # ==================================================================================================
    g, adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    in_feats = features.shape[1]

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    GAT = GATNodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop).to(device)
    GAT_LN = GATNodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop, layer_norm=True).to(device)
    GAT_AT = GATNodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop).to(device)
    GAT_FGAI = GATNodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop).to(device)
    with open(f"./GATv2/optimized_hyperparameter_configurations/FGAI_{dataset}.yml", 'r') as file:
        args = yaml.safe_load(file)
    args = argparse.Namespace(**args)
    args.device = device
    GATv2 = GATv2NodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop
    ).to(device)
    GATv2_AT = GATv2NodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop
    ).to(device)
    GATv2_LN = GATv2NodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop, layer_norm=True
    ).to(device)
    GATv2_FGAI = GATv2NodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop
    ).to(device)
    with open(f"./GT/optimized_hyperparameter_configurations/FGAI_{dataset}.yml", 'r') as file:
        args = yaml.safe_load(file)
    args = argparse.Namespace(**args)
    args.device = device
    pos_enc_size = 8
    args.hid_dim = 80
    GT = GTNodeClassifier(
        feats_size=features.shape[1],
        hidden_size=args.hid_dim,
        out_size=num_classes,
        pos_enc_size=pos_enc_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)
    GT_LN = GTNodeClassifier(
        feats_size=features.shape[1],
        hidden_size=args.hid_dim,
        out_size=num_classes,
        pos_enc_size=pos_enc_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads, layer_norm=True
    ).to(device)
    GT_AT = GTNodeClassifier(
        feats_size=features.shape[1],
        hidden_size=args.hid_dim,
        out_size=num_classes,
        pos_enc_size=pos_enc_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)
    GT_FGAI = GTNodeClassifier(
        feats_size=features.shape[1],
        hidden_size=args.hid_dim,
        out_size=num_classes,
        pos_enc_size=pos_enc_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)

    # ==================================================================================================
    # 6. Load pre-trained vanilla model
    # ==================================================================================================
    # tim1 = '11-52'
    # tim_AT = '11-47'
    # tim_LN = '11-28'
    # tim_FGAI = '01-23_12-01'
    # tim1 = '13-55'
    # tim_AT = '11-55'
    # tim_LN = '19-36'
    # tim_FGAI = '01-24_21-27'
    # tim1 = '21-09'
    # tim_AT = '12-34'
    # tim_LN = '21-02'
    # tim_FGAI = '01-23_21-52'
    tim1 = '14-02'
    tim_AT = '13-07'
    tim_LN = '20-52'
    tim_FGAI = '01-25_08-41'
    GAT.load_state_dict(torch.load(f'./GAT/vanilla_checkpoints/{dataset}_{tim1}/model_parameters.pth'))
    GAT_AT.load_state_dict(torch.load(f'./GAT/AT_checkpoints/{dataset}_{tim_AT}/model_parameters.pth'))
    GAT_LN.load_state_dict(torch.load(f'./GAT/LN_checkpoints/{dataset}_{tim_LN}/model_parameters.pth'))
    GAT_FGAI.load_state_dict(torch.load(f'./GAT/FGAI_checkpoints/{dataset}_{tim_FGAI}/FGAI_parameters.pth'))

    # tim2 = '16-37'
    # tim_AT = '19-43'
    # tim_LN = '12-31'
    # tim_FGAI = '01-23_18-21'
    # tim2 = '22-29'
    # tim_AT = '19-54'
    # tim_LN = '20-11'
    # tim_FGAI = '01-25_19-44'
    # tim2 = '22-21'
    # tim_AT = '20-14'
    # tim_LN = '20-26'
    # tim_FGAI = '01-26_19-10'
    tim2 = '22-35'
    tim_AT = '17-09'
    tim_LN = '20-39'
    tim_FGAI = '01-26_10-41'
    GATv2.load_state_dict(torch.load(f'./GATv2/vanilla_checkpoints/{dataset}_{tim2}/model_parameters.pth'))
    GATv2_AT.load_state_dict(torch.load(f'./GATv2/AT_checkpoints/{dataset}_{tim_AT}/model_parameters.pth'))
    GATv2_LN.load_state_dict(torch.load(f'./GATv2/LN_checkpoints/{dataset}_{tim_LN}/model_parameters.pth'))
    GATv2_FGAI.load_state_dict(torch.load(f'./GATv2/FGAI_checkpoints/{dataset}_{tim_FGAI}/FGAI_parameters.pth'))

    # tim3 = '19-35'
    # tim_AT = '20-37'
    # tim_LN = '14-54'
    # tim_FGAI = '01-26_22-58'
    # tim3 = '19-40'
    # tim_AT = '20-59'
    # tim_LN = '15-16'
    # tim_FGAI = '01-26_23-25'
    # tim3 = '23-40'
    # tim_AT = '21-16'
    # tim_LN = '15-30'
    # tim_FGAI = '01-27_00-15'
    tim3 = '12-26'
    tim_AT = '12-42'
    tim_LN = '12-36'
    tim_FGAI = '01-30_13-18'
    GT.load_state_dict(torch.load(f'./GT/vanilla_checkpoints/{dataset}_{tim3}/model_parameters.pth'))
    GT_AT.load_state_dict(torch.load(f'./GT/AT_checkpoints/{dataset}_{tim_AT}/model_parameters.pth'))
    GT_LN.load_state_dict(torch.load(f'./GT/LN_checkpoints/{dataset}_{tim_LN}/model_parameters.pth'))
    GT_FGAI.load_state_dict(torch.load(f'./GT/FGAI_checkpoints/{dataset}_{tim_FGAI}/FGAI_parameters.pth'))
    GT.pos_enc = torch.load(f'./GT/{dataset}_pos_enc.pth').to(device)
    GT.pos_enc_ = torch.load(f'./GT/{dataset}_pos_enc_perturbed.pth').to(device)
    GT_AT.pos_enc = torch.load(f'./GT/{dataset}_pos_enc.pth').to(device)
    GT_AT.pos_enc_ = torch.load(f'./GT/{dataset}_pos_enc_perturbed.pth').to(device)
    GT_LN.pos_enc = torch.load(f'./GT/{dataset}_pos_enc.pth').to(device)
    GT_LN.pos_enc_ = torch.load(f'./GT/{dataset}_pos_enc_perturbed.pth').to(device)
    GT_FGAI.pos_enc = torch.load(f'./GT/{dataset}_pos_enc.pth').to(device)
    GT_FGAI.pos_enc_ = torch.load(f'./GT/{dataset}_pos_enc_perturbed.pth').to(device)

    # ==================================================================================================
    # 8. Evaluation
    # ==================================================================================================
    adj_ = sp.load_npz(f'./GAT/vanilla_checkpoints/{dataset}_{tim1}/adj_delta.npz')
    feats_perturbed = torch.load(f'./GAT/vanilla_checkpoints/{dataset}_{tim1}/feats_delta.pth').to(device)
    feats_ = torch.cat((features, feats_perturbed), dim=0)


    def compute_fidelity(model, adj, feats, adj_, feats_, labels, test_idx):
        model.eval()
        outputs, _, att = evaluate_node_level(model, feats, adj, label, test_idx)
        _, _, attention = evaluate_node_level(model, feats_, adj_, label, test_idx)
        if len(attention.shape) > 1:
            attention = torch.mean(attention, dim=1)
        # attention = attention[:att.shape[0]]
        pred = torch.argmax(outputs, dim=1)[test_idx]
        labels_test = labels[test_idx]
        corr_idx = torch.where(pred == labels_test)[0]
        rows, cols = adj_.nonzero()
        shape = adj_.shape
        sorted_indices = torch.argsort(attention, descending=True).cpu().numpy()

        fidelity_pos_list, fidelity_neg_list = [], []
        for split in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            num_edges_to_keep = int(split * len(attention))
            edges_to_keep = sorted_indices[:num_edges_to_keep]
            rows_to_keep = rows[edges_to_keep]
            cols_to_keep = cols[edges_to_keep]
            data = np.ones(len(rows_to_keep))
            adj_imp = csr_matrix((data, (rows_to_keep, cols_to_keep)), shape)

            edges_to_keep = sorted_indices[-num_edges_to_keep:]
            rows_to_keep = rows[edges_to_keep]
            cols_to_keep = cols[edges_to_keep]
            data = np.ones(len(rows_to_keep))
            adj_unimp = csr_matrix((data, (rows_to_keep, cols_to_keep)), shape)

            outputs_wo_imp, _, _ = model(feats_, adj_unimp)
            outputs_wo_unimp, _, _ = model(feats_, adj_imp)

            pred_wo_imp = torch.argmax(outputs_wo_imp, dim=1)[test_idx]
            pred_wo_unimp = torch.argmax(outputs_wo_unimp, dim=1)[test_idx]

            fidelity_pos = torch.sum(pred_wo_unimp[corr_idx] == labels_test[corr_idx]) / len(corr_idx)
            fidelity_neg = torch.sum(pred_wo_imp[corr_idx] == labels_test[corr_idx]) / len(corr_idx)
            fidelity_pos_list.append(round(fidelity_pos.item(), 4))
            fidelity_neg_list.append(round(fidelity_neg.item(), 4))

        return fidelity_pos_list, fidelity_neg_list


    dataset_ = dataset.replace('_', '-')

    print("GAT")
    f_pos_list, f_neg_list = compute_fidelity(GAT, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GAT.txt'), sep=',', index=False)

    print("GAT+LN")
    f_pos_list, f_neg_list = compute_fidelity(GAT_LN, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GAT+LN.txt'), sep=',', index=False)

    print("GAT+AT")
    f_pos_list, f_neg_list = compute_fidelity(GAT_AT, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GAT+AT.txt'), sep=',', index=False)

    print("GAT+FGAI")
    f_pos_list, f_neg_list = compute_fidelity(GAT_FGAI, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GAT+FGAI.txt'), sep=',', index=False)

    adj_ = sp.load_npz(f'./GATv2/vanilla_checkpoints/{dataset}_{tim2}/adj_delta.npz')
    feats_perturbed = torch.load(f'./GATv2/vanilla_checkpoints/{dataset}_{tim2}/feats_delta.pth').to(device)
    feats_ = torch.cat((features, feats_perturbed), dim=0)

    print("GATv2")
    f_pos_list, f_neg_list = compute_fidelity(GATv2, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GATv2.txt'), sep=',', index=False)

    print("GATv2+LN")
    f_pos_list, f_neg_list = compute_fidelity(GATv2_LN, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GATv2+LN.txt'), sep=',', index=False)

    print("GATv2+AT")
    f_pos_list, f_neg_list = compute_fidelity(GATv2_AT, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GATv2+AT.txt'), sep=',', index=False)

    print("GATv2+FGAI")
    f_pos_list, f_neg_list = compute_fidelity(GATv2_FGAI, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GATv2+FGAI.txt'), sep=',', index=False)

    adj_ = sp.load_npz(f'./GT/vanilla_checkpoints/{dataset}_{tim3}/adj_delta.npz')
    feats_perturbed = torch.load(f'./GT/vanilla_checkpoints/{dataset}_{tim3}/feats_delta.pth').to(device)
    feats_ = torch.cat((features, feats_perturbed), dim=0)

    print("GT")
    f_pos_list, f_neg_list = compute_fidelity(GT, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GT.txt'), sep=',', index=False)

    print("GT+LN")
    f_pos_list, f_neg_list = compute_fidelity(GT_LN, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GT+LN.txt'), sep=',', index=False)

    print("GT+AT")
    f_pos_list, f_neg_list = compute_fidelity(GT_AT, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GT+AT.txt'), sep=',', index=False)

    print("GT+FGAI")
    f_pos_list, f_neg_list = compute_fidelity(GT_FGAI, adj, features, adj_, feats_, label, test_idx)
    print(f"fidelity_pos: {f_pos_list}")
    print(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(f'./F_after_attack/{dataset_}', f'GT+FGAI.txt'), sep=',', index=False)
