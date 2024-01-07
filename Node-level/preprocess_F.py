import os
import yaml
import torch
import argparse
import pandas as pd
import scipy.sparse as sp
from models import GATNodeClassifier, GATv2NodeClassifier
from load_dataset import load_dataset


def compute_fidelity(model, adj, feats, labels, test_idx, num_nodes):
    model.eval()

    variances = torch.var(feats, dim=0)
    sorted_indices = torch.argsort(variances)

    fidelity_pos_list, fidelity_neg_list = [], []
    for split in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        imp_indices = sorted_indices[-int(feats.shape[1] * split):]
        unimp_indices = sorted_indices[:int(feats.shape[1] * split)]
        feats_imp = torch.zeros_like(feats)
        feats_imp[:, imp_indices] = feats[:, imp_indices]
        feats_unimp = torch.zeros_like(feats)
        feats_unimp[:, unimp_indices] = feats[:, unimp_indices]

        outputs_wo_imp, _, _ = model(feats_unimp, adj)
        outputs_wo_unimp, _, _ = model(feats_imp, adj)
        outputs, _, _ = model(feats, adj)

        outputs_wo_imp = outputs_wo_imp[:num_nodes]
        outputs_wo_unimp = outputs_wo_unimp[:num_nodes]
        outputs = outputs[:num_nodes]

        pred_wo_imp = torch.argmax(outputs_wo_imp, dim=1)[test_idx]
        pred_wo_unimp = torch.argmax(outputs_wo_unimp, dim=1)[test_idx]
        pred = torch.argmax(outputs, dim=1)[test_idx]
        labels_test = labels[test_idx]

        # fidelity_pos = torch.sum(pred_wo_unimp == labels_test) / len(labels_test)
        # fidelity_neg = torch.sum(pred_wo_imp == labels_test) / len(labels_test)
        # fidelity_pos_list.append(round(fidelity_pos.item(), 4))
        # fidelity_neg_list.append(round(fidelity_neg.item(), 4))
        corr_idx = torch.where(pred == labels_test)[0]
        fidelity_pos = torch.sum(pred_wo_unimp[corr_idx] == labels_test[corr_idx]) / len(corr_idx)
        fidelity_neg = torch.sum(pred_wo_imp[corr_idx] == labels_test[corr_idx]) / len(corr_idx)
        fidelity_pos_list.append(round(fidelity_pos.item(), 4))
        fidelity_neg_list.append(round(fidelity_neg.item(), 4))

    print(torch.sum(pred == labels_test) / len(labels_test))
    return fidelity_pos_list, fidelity_neg_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

if __name__ == '__main__':
    exp = 'GAT+AT'
    dataset = 'amazon_photo'
    # dataset = 'amazon_cs'
    # dataset = 'pubmed'
    # dataset = 'coauthor_phy'
    # dataset ='ogbn-arxiv'

    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment args ' + '=' * 36)

    with open(f"./{exp}/optimized_hyperparameter_configurations/FGAI_{dataset}.yml", 'r') as file:
        args = yaml.safe_load(file)
    args = argparse.Namespace(**args)
    args.device = device

    # ==================================================================================================
    # 4. Prepare data
    # ==================================================================================================
    adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    in_feats = features.shape[1]

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    vanilla_model = GATNodeClassifier(feats_size=in_feats,
                                      hidden_size=args.hid_dim,
                                      out_size=num_classes,
                                      n_layers=args.n_layers,
                                      n_heads=args.n_heads,
                                      feat_drop=args.feat_drop,
                                      attn_drop=args.attn_drop).to(device)
    FGAI = GATNodeClassifier(feats_size=in_feats,
                             hidden_size=args.hid_dim,
                             out_size=num_classes,
                             n_layers=args.n_layers,
                             n_heads=args.n_heads,
                             feat_drop=args.feat_drop,
                             attn_drop=args.attn_drop).to(device)

    # ==================================================================================================
    # 6. Load pre-trained vanilla model
    # ==================================================================================================
    tim = '19-06'
    tim_FGAI = '10-07_23-03'
    vanilla_model.load_state_dict(torch.load(f'./{exp}/GAT_checkpoints/{dataset}_{tim}/model_parameters.pth'))
    FGAI.load_state_dict(torch.load(f'./{exp}/FGAI_checkpoints/{dataset}_{tim_FGAI}/FGAI_parameters.pth'))

    # ==================================================================================================
    # 8. Evaluation
    # ==================================================================================================
    adj_perturbed = sp.load_npz(f'./{exp}/GAT_checkpoints/{args.dataset}_{tim}/adj_delta.npz')
    feats_perturbed = torch.load(f'./{exp}/GAT_checkpoints/{args.dataset}_{tim}/feats_delta.pth').to(device)
    features_perturbed = torch.cat((features, feats_perturbed), dim=0)

    fidelity_pos_list, fidelity_neg_list = compute_fidelity(vanilla_model, adj_perturbed, features_perturbed, label,
                                                            test_idx, len(features))
    fidelity_pos_list_FGAI, fidelity_neg_list_FGAI = compute_fidelity(FGAI, adj_perturbed, features_perturbed, label,
                                                                      test_idx, len(features))

    dataset = 'amazon-photo'
    # dataset = 'amazon-cs'
    # dataset = 'pubmed'
    # dataset = 'coauthor-phy'
    # dataset ='ogbn-arxiv'

    print(f"fidelity_pos: {fidelity_pos_list}")
    print(f"fidelity_neg: {fidelity_neg_list}")
    data = pd.DataFrame({'fidelity_pos': fidelity_pos_list, 'fidelity_neg': fidelity_neg_list})
    data.to_csv(os.path.join(f'./F/{dataset}', f'{exp}.txt'), sep=',', index=False)

    print(f"fidelity_pos FGAI: {fidelity_pos_list_FGAI}")
    print(f"fidelity_neg FGAI: {fidelity_neg_list_FGAI}")
    data = pd.DataFrame({'fidelity_pos': fidelity_pos_list_FGAI, 'fidelity_neg': fidelity_neg_list_FGAI})
    data.to_csv(os.path.join(f'./F/{dataset}', f'{exp}_FGAI.txt'), sep=',', index=False)
