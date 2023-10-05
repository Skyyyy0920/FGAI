import yaml
import time
import zipfile
import argparse
import pandas as pd
from pathlib import Path
from models import GATNodeClassifier
from utils import *
from trainer import FGAITrainer
from load_dataset import load_dataset
from attackers import PGD
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = 'cpu'


if __name__ == '__main__':
    # dataset ='ogbn-arxiv'
    # dataset='ogbn-products'
    # dataset='ogbn-papers100M'
    dataset = 'pubmed'
    # dataset='questions'
    # dataset='amazon-ratings'
    # dataset='roman-empire'
    # dataset='amazon_photo'
    # dataset='amazon_cs'
    # dataset='coauthor_cs'
    # dataset='coauthor_phy'

    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment args ' + '=' * 36)

    with open(f"./optimized_hyperparameter_configurations/FGAI_{dataset}.yml", 'r') as file:
        args = yaml.safe_load(file)
    args = argparse.Namespace(**args)
    args.device = device

    # ==================================================================================================
    # 2. Setup logger
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Setup logger ' + '=' * 36)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join("./FGAI_checkpoints/", f"{dataset}_{logging_time}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'{dataset}.log'))
    console = logging.StreamHandler()  # Simultaneously output to console
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    logging.info(f"Using device: {device}")
    logging.info(f"PyTorch Version: {torch.__version__}")
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")

    # ==================================================================================================
    # 3. Save codes and settings
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Save codes and settings ' + '=' * 36)
    zipf = zipfile.ZipFile(file=os.path.join(save_dir, 'codes.zip'), mode='a', compression=zipfile.ZIP_DEFLATED)
    zipdir(Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    with open(os.path.join(save_dir, 'args.yml'), 'a') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # ==================================================================================================
    # 4. Prepare data
    # ==================================================================================================
    adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    in_feats = features.shape[1]

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    vanilla_model = GATNodeClassifier(in_feats=in_feats,
                                      hid_dim=args.hid_dim,
                                      n_classes=num_classes,
                                      n_layers=args.n_layers,
                                      n_heads=args.n_heads,
                                      feat_drop=args.feat_drop,
                                      attn_drop=args.attn_drop).to(device)
    FGAI = GATNodeClassifier(in_feats=in_feats,
                             hid_dim=args.hid_dim,
                             n_classes=num_classes,
                             n_layers=args.n_layers,
                             n_heads=args.n_heads,
                             feat_drop=args.feat_drop,
                             attn_drop=args.attn_drop).to(device)
    optimizer = optim.Adam(FGAI.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    logging.info(f"Model: {FGAI}")
    logging.info(f"Optimizer: {optimizer}")

    attacker_delta = PGD(epsilon=args.epsilon,
                         n_epoch=args.n_epoch_attack,
                         n_inject_max=args.n_inject_max,
                         n_edge_max=args.n_edge_max,
                         feat_lim_min=features.min().item(),
                         feat_lim_max=features.max().item(),
                         loss=TVD,
                         device=device)
    attacker_rho = PGD(epsilon=args.epsilon,
                       n_epoch=args.n_epoch_attack,
                       n_inject_max=args.n_inject_max,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=features.min().item(),
                       feat_lim_max=features.max().item(),
                       loss=topK_overlap_loss,
                       K=args.K_rho,
                       device=device)

    trainer = FGAITrainer(FGAI, optimizer, attacker_delta, attacker_rho, args)

    # ==================================================================================================
    # 6. Load pre-trained vanilla model
    # ==================================================================================================
    tim = '_00-27'
    vanilla_model.load_state_dict(torch.load(f'./GAT_checkpoints/{dataset}{tim}/model_parameters.pth'))

    orig_outputs, orig_graph_repr, orig_att = \
        evaluate_node_level(vanilla_model, features, adj, label, test_idx, num_classes == 2)

    # ==================================================================================================
    # 7. Train our FGAI
    # ==================================================================================================
    idx_split = train_idx, valid_idx, test_idx
    trainer.train(features, adj, label, idx_split, orig_outputs, orig_graph_repr, orig_att)

    FGAI_outputs, FGAI_graph_repr, FGAI_att = \
        evaluate_node_level(FGAI, features, adj, label, test_idx, num_classes == 2)

    # ==================================================================================================
    # 7. Save FGAI
    # ==================================================================================================
    torch.save(FGAI.state_dict(), f'{save_dir}/FGAI_parameters.pth')

    # ==================================================================================================
    # 8. Evaluation
    # ==================================================================================================
    adj_perturbed = sp.load_npz(f'./GAT_checkpoints/{args.dataset}{tim}/adj_delta.npz')
    feats_perturbed = torch.load(f'./GAT_checkpoints/{args.dataset}{tim}/feats_delta.pth').to(device)

    FGAI.eval()
    new_outputs, new_graph_repr, new_att = FGAI(torch.cat((features, feats_perturbed), dim=0), adj_perturbed)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:FGAI_outputs.shape[0]], new_graph_repr[:FGAI_graph_repr.shape[0]], new_att[:FGAI_att.shape[0]]

    TVD_score = TVD(FGAI_att, new_att) / len(new_att)
    JSD_score = JSD(FGAI_att, new_att) / len(new_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    fidelity_pos_list, fidelity_neg_list = compute_fidelity(FGAI, adj, features, label, test_idx)
    logging.info(f"fidelity_pos: {fidelity_pos_list}")
    logging.info(f"fidelity_neg: {fidelity_neg_list}")
    data = pd.DataFrame({'fidelity_pos': fidelity_pos_list, 'fidelity_neg': fidelity_neg_list})
    data.to_csv(os.path.join(save_dir, 'fidelity_data.txt'), sep=',', index=False)
