import yaml
import time
import torch.nn as nn
import zipfile
import argparse
from pathlib import Path

from models import GATNodeClassifier, GATv2NodeClassifier
from utils import *
from trainer import FGAITrainer
from load_dataset import load_dataset
from attackers import PGD
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


def get_args():
    parser = argparse.ArgumentParser(description="MTNet's args")

    # Operation environment
    parser.add_argument('--seed', type=int, default=20010920, help='Random seed')
    parser.add_argument('--device', type=str, default=device, help='Running on which device')

    # Data
    parser.add_argument('--dataset',
                        type=str,
                        # default='ogbn-arxiv',
                        # default='ogbn-products',
                        # default='ogbn-papers100M',
                        # default='pubmed',
                        default='questions',
                        # default='amazon-ratings',
                        # default='roman-empire',
                        help='Dataset name')

    # Experimental Setup
    parser.add_argument('--num_epochs', type=int, default=100, help='Training epoch')
    parser.add_argument('--early_stopping', type=int, default=6)

    parser.add_argument('--n_inject_max', type=int, default=20)
    parser.add_argument('--n_edge_max', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--n_epoch_attack', type=int, default=10)

    parser.add_argument('--lambda_1', type=float, default=5e-2)
    parser.add_argument('--lambda_2', type=float, default=1e-2)
    parser.add_argument('--lambda_3', type=float, default=2e-2)
    parser.add_argument('--K', type=int, default=200000)
    parser.add_argument('--K_rho', type=int, default=200000)

    parser.add_argument('--save_path', type=str, default='./FGAI_checkpoints/', help='Checkpoints saving path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment args ' + '=' * 36)
    args = get_args()
    # setup_seed(args.seed)  # make the experiment repeatable

    # ==================================================================================================
    # 2. Setup logger
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Setup logger ' + '=' * 36)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_{logging_time}")
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

    logging.info(f"Using device: {args.device}")
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
    g, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    features = g.ndata["feat"]
    in_feats = features.shape[1]
    src, dst = g.edges()
    num_nodes = g.number_of_nodes()
    adj = sp.csr_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())), shape=(num_nodes, num_nodes))
    del g
    logging.info(f"num_nodes: {num_nodes}")

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    if args.dataset == 'ogbn-arxiv':
        vanilla_model = GATNodeClassifier(in_feats=in_feats,
                                          hid_dim=128,
                                          n_classes=num_classes,
                                          n_layers=3,
                                          n_heads=[4, 2, 1],
                                          feat_drop=0.05,
                                          attn_drop=0).to(args.device)
        FGAI = GATNodeClassifier(in_feats=in_feats,
                                 hid_dim=128,
                                 n_classes=num_classes,
                                 n_layers=3,
                                 n_heads=[4, 2, 1],
                                 feat_drop=0.05,
                                 attn_drop=0).to(args.device)
        optimizer = optim.Adam(FGAI.parameters(),
                               lr=1e-2,
                               weight_decay=0)
    elif args.dataset in ['cora', 'citeseer']:
        vanilla_model = GATNodeClassifier(in_feats=in_feats,
                                          hid_dim=8,
                                          n_classes=num_classes,
                                          n_layers=1,
                                          n_heads=[8],
                                          feat_drop=0.6,
                                          attn_drop=0.6).to(args.device)
        FGAI = GATNodeClassifier(in_feats=in_feats,
                                 hid_dim=8,
                                 n_classes=num_classes,
                                 n_layers=1,
                                 n_heads=[8],
                                 feat_drop=0.6,
                                 attn_drop=0.6).to(args.device)
        optimizer = optim.Adam(FGAI.parameters(),
                               lr=1e-2,
                               weight_decay=5e-4)
    elif args.dataset == 'pubmed':
        vanilla_model = GATNodeClassifier(in_feats=in_feats,
                                          hid_dim=8,
                                          n_classes=num_classes,
                                          n_layers=1,
                                          n_heads=[8],
                                          feat_drop=0.6,
                                          attn_drop=0.6).to(args.device)
        FGAI = GATNodeClassifier(in_feats=in_feats,
                                 hid_dim=8,
                                 n_classes=num_classes,
                                 n_layers=1,
                                 n_heads=[8],
                                 feat_drop=0.6,
                                 attn_drop=0.6).to(args.device)
        optimizer = optim.Adam(FGAI.parameters(),
                               lr=1e-2,
                               weight_decay=5e-4)
    elif args.dataset == 'amazon-ratings':
        args.num_epochs = 200
        args.early_stopping = 200
        vanilla_model = GATNodeClassifier(in_feats=in_feats,
                                            hid_dim=128,
                                            n_classes=num_classes,
                                            n_layers=2,
                                            n_heads=[8, 4],
                                            feat_drop=0,
                                            attn_drop=0).to(args.device)
        FGAI = GATNodeClassifier(in_feats=in_feats,
                                   hid_dim=128,
                                   n_classes=num_classes,
                                   n_layers=2,
                                   n_heads=[8, 4],
                                   feat_drop=0,
                                   attn_drop=0).to(args.device)
        optimizer = optim.Adam(FGAI.parameters(),
                               lr=1e-3,
                               weight_decay=5e-4)
    elif args.dataset == 'questions':
        # args.num_epochs = 150
        vanilla_model = GATNodeClassifier(in_feats=in_feats,
                                          hid_dim=128,
                                          n_classes=num_classes,
                                          n_layers=2,
                                          n_heads=[8, 4],
                                          feat_drop=0,
                                          attn_drop=0).to(args.device)
        FGAI = GATNodeClassifier(in_feats=in_feats,
                                 hid_dim=128,
                                 n_classes=num_classes,
                                 n_layers=2,
                                 n_heads=[8, 4],
                                 feat_drop=0,
                                 attn_drop=0).to(args.device)
        optimizer = optim.Adam(FGAI.parameters(),
                               lr=1e-2,
                               weight_decay=5e-4)
    else:  # default='roman-empire',
        # args.num_epochs = 400
        vanilla_model = GATNodeClassifier(in_feats=in_feats,
                                          hid_dim=128,
                                          n_classes=num_classes,
                                          n_layers=2,
                                          n_heads=[8, 4],
                                          feat_drop=0,
                                          attn_drop=0).to(args.device)
        FGAI = GATNodeClassifier(in_feats=in_feats,
                                 hid_dim=128,
                                 n_classes=num_classes,
                                 n_layers=2,
                                 n_heads=[8, 4],
                                 feat_drop=0,
                                 attn_drop=0).to(args.device)
        optimizer = optim.Adam(FGAI.parameters(),
                               lr=1e-2,
                               weight_decay=5e-4)

    logging.info(f"Model: {FGAI}")
    logging.info(f"Optimizer: {optimizer}")

    attacker_delta = PGD(epsilon=args.epsilon,
                         n_epoch=args.n_epoch_attack,
                         n_inject_max=args.n_inject_max,
                         n_edge_max=args.n_edge_max,
                         feat_lim_min=features.min().item(),
                         feat_lim_max=features.max().item(),
                         loss=TVD,
                         device=args.device)
    attacker_rho = PGD(epsilon=args.epsilon,
                       n_epoch=args.n_epoch_attack,
                       n_inject_max=args.n_inject_max,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=features.min().item(),
                       feat_lim_max=features.max().item(),
                       loss=topK_overlap_loss,
                       K=args.K_rho,
                       device=args.device)
    criterion = nn.CrossEntropyLoss()

    FGAI_trainer = FGAITrainer(FGAI, optimizer, attacker_delta, attacker_rho, args)

    # ==================================================================================================
    # 6. Load pre-trained vanilla model
    # ==================================================================================================
    tim = '_02-12'
    vanilla_model.load_state_dict(torch.load(f'./vanilla_model/{args.dataset}{tim}/model_parameters.pth'))
    vanilla_model.eval()

    tensor_dict = torch.load(f'./vanilla_model/{args.dataset}{tim}/orig_tensors.pth')
    orig_outputs = tensor_dict['orig_outputs'].to(device=args.device)
    orig_graph_repr = tensor_dict['orig_graph_repr'].to(device=args.device)
    orig_att = tensor_dict['orig_att'].to(device=args.device)

    evaluate_node_level(vanilla_model, criterion, features, adj, label, test_idx)

    # ==================================================================================================
    # 7. Train our FGAI
    # ==================================================================================================
    idx_split = train_idx, valid_idx, test_idx
    FGAI_outputs, FGAI_graph_repr, FGAI_att = FGAI_trainer.train(features, adj, label, idx_split, orig_outputs,
                                                                 orig_graph_repr, orig_att)
    evaluate_node_level(FGAI, criterion, features, adj, label, test_idx)

    # ==================================================================================================
    # 7. Save FGAI
    # ==================================================================================================
    torch.save(FGAI.state_dict(), f'{save_dir}/FGAI_parameters.pth')

    # ==================================================================================================
    # 8. Evaluation
    # ==================================================================================================
    adj_perturbed = sp.load_npz(f'./vanilla_model/{args.dataset}{tim}/adj_delta.npz')
    feats_perturbed = torch.load(f'./vanilla_model/{args.dataset}{tim}/feats_delta.pth').to(features.device)

    new_outputs, new_graph_repr, new_att = FGAI(torch.cat((features, feats_perturbed), dim=0), adj_perturbed)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:FGAI_outputs.shape[0]], new_graph_repr[:FGAI_graph_repr.shape[0]], new_att[:FGAI_att.shape[0]]

    TVD_score = TVD(FGAI_att, new_att) / len(orig_att)
    JSD_score = JSD(FGAI_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    fidelity_pos, fidelity_neg = compute_fidelity(FGAI, adj, features, label, test_idx)
    logging.info(f"fidelity_pos: {fidelity_pos}, fidelity_neg: {fidelity_neg}")
