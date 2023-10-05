import yaml
import time
import torch.nn as nn
import zipfile
import argparse
from pathlib import Path
from models import GATNodeClassifier, GNNGuard
from utils import *
from trainer import FGAITrainer
from load_dataset import load_dataset
from attackers import PGD
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = 'cpu'


def get_args():
    parser = argparse.ArgumentParser(description="FGAI's args")

    # Data
    parser.add_argument('--dataset',
                        type=str,
                        # default='ogbn-arxiv',
                        # default='ogbn-products',
                        # default='ogbn-papers100M',
                        # default='pubmed',
                        # default='questions',
                        # default='amazon-ratings',
                        # default='roman-empire',
                        default='amazon_photo',
                        # default='amazon_cs',
                        # default='coauthor_cs',
                        # default='coauthor_phy',
                        help='Dataset name')

    # Experimental Setup
    parser.add_argument('--num_epochs', type=int, default=100, help='Training epoch')
    parser.add_argument('--early_stopping', type=int, default=6)

    parser.add_argument('--n_inject_max', type=int, default=20)
    parser.add_argument('--n_edge_max', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--n_epoch_attack', type=int, default=10)

    parser.add_argument('--hid_dim', type=int, default=8)
    parser.add_argument('--n_heads', type=list, default=[8])
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--feat_drop', type=float, default=0.05)
    parser.add_argument('--attn_drop', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--lambda_1', type=float, default=5e-2)
    parser.add_argument('--lambda_2', type=float, default=1e-2)
    parser.add_argument('--lambda_3', type=float, default=2e-2)
    parser.add_argument('--K', type=int, default=200000)
    parser.add_argument('--K_rho', type=int, default=200000)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment args ' + '=' * 36)
    args = get_args()
    load_optimized_hyperparameter_configurations = True
    if load_optimized_hyperparameter_configurations:
        with open(f"./optimized_hyperparameter_configurations/FGAI_{args.dataset}.yml", 'r') as file:
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
    save_dir = os.path.join("./FGAI_checkpoints/", f"{args.dataset}_{logging_time}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving path: {save_dir}")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'{args.dataset}.log'))
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
    # GAT_checkpoints = GATNodeClassifier(in_feats=in_feats,
    #                                   hid_dim=args.hid_dim,
    #                                   n_classes=num_classes,
    #                                   n_layers=args.n_layers,
    #                                   n_heads=args.n_heads,
    #                                   feat_drop=args.feat_drop,
    #                                   attn_drop=args.attn_drop).to(device)
    # FGAI = GATNodeClassifier(in_feats=in_feats,
    #                          hid_dim=args.hid_dim,
    #                          n_classes=num_classes,
    #                          n_layers=args.n_layers,
    #                          n_heads=args.n_heads,
    #                          feat_drop=args.feat_drop,
    #                          attn_drop=args.attn_drop).to(device)
    FGAI = GNNGuard(in_feats=in_feats,
                             hid_dim=args.hid_dim,
                             n_classes=num_classes,
                             n_layers=args.n_layers,
                             n_heads=args.n_heads,
                             dropout=0.6).to(device)
    vanilla_model = GNNGuard(in_feats=in_feats,
                             hid_dim=args.hid_dim,
                             n_classes=num_classes,
                             n_layers=args.n_layers,
                             n_heads=args.n_heads,
                             dropout=0.6).to(device)
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
    criterion = nn.CrossEntropyLoss()

    trainer = FGAITrainer(FGAI, optimizer, attacker_delta, attacker_rho, args)

    # ==================================================================================================
    # 6. Load pre-trained vanilla model
    # ==================================================================================================
    tim = '_19-44'
    # GAT_checkpoints.load_state_dict(torch.load(f'./GAT_checkpoints/{args.dataset}{tim}/model_parameters.pth'))
    vanilla_model.load_state_dict(torch.load(f'./GNNGuard_checkpoints/amazon_photo_18-23/model_parameters.pth'))
    vanilla_model.eval()

    # tensor_dict = torch.load(f'./GAT_checkpoints/{args.dataset}{tim}/orig_tensors.pth')
    tensor_dict = torch.load(f'./GNNGuard_checkpoints/amazon_photo_18-23/tensors.pth')
    orig_outputs = tensor_dict['orig_outputs'].to(device=device)
    orig_graph_repr = tensor_dict['orig_graph_repr'].to(device=device)
    orig_att = tensor_dict['orig_att'].to(device=device)

    evaluate_node_level(vanilla_model, criterion, features, adj, label, test_idx, num_classes == 2)

    # ==================================================================================================
    # 7. Train our FGAI
    # ==================================================================================================
    idx_split = train_idx, valid_idx, test_idx
    FGAI_outputs, FGAI_graph_repr, FGAI_att = trainer.train(features, adj, label, idx_split, orig_outputs,
                                                            orig_graph_repr, orig_att)
    evaluate_node_level(FGAI, criterion, features, adj, label, test_idx, num_classes == 2)

    # ==================================================================================================
    # 7. Save FGAI
    # ==================================================================================================
    torch.save(FGAI.state_dict(), f'{save_dir}/FGAI_parameters.pth')

    # ==================================================================================================
    # 8. Evaluation
    # ==================================================================================================
    adj_perturbed = sp.load_npz(f'./vanilla_model/{args.dataset}{tim}/adj_delta.npz')
    feats_perturbed = torch.load(f'./vanilla_model/{args.dataset}{tim}/feats_delta.pth').to(device)

    new_outputs, new_graph_repr, new_att = FGAI(torch.cat((features, feats_perturbed), dim=0), adj_perturbed)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:FGAI_outputs.shape[0]], new_graph_repr[:FGAI_graph_repr.shape[0]], new_att[:FGAI_att.shape[0]]

    TVD_score = TVD(FGAI_att, new_att) / len(orig_att)
    JSD_score = JSD(FGAI_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    fidelity_pos, fidelity_neg = compute_fidelity(FGAI, adj, features, label, test_idx)
    logging.info(f"fidelity_pos: {fidelity_pos}, fidelity_neg: {fidelity_neg}")
