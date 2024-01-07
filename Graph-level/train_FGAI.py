import yaml
import time
import torch.nn as nn
import zipfile
import argparse
from pathlib import Path

from models import GATGraphClassifier
from utils import *
from trainer import FGAITrainer
from load_dataset import load_dataset
from attackers import PGD
import torch.optim as optim
from grb.attack.fgsm import FGSM
from grb.attack.tdgia import TDGIA
from grb.attack.rnd import RND
from grb.attack.speit import SPEIT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description="MTNet's args")

    # Operation environment
    parser.add_argument('--seed', type=int, default=20010920, help='Random seed')
    parser.add_argument('--device', type=str, default=device, help='Running on which device')

    # Data
    parser.add_argument('--task', type=str, default='node-level', help='task')  # default='graph-level'
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-arxiv',
                        help='Dataset name')

    # Experimental Setup
    parser.add_argument('--num_epochs', type=int, default=300, help='Training epoch')

    parser.add_argument('--n_inject_max', type=int, default=50)
    parser.add_argument('--n_edge_max', type=int, default=50)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--n_epoch_attack', type=int, default=10)

    parser.add_argument('--lambda_1', type=float, default=1e-2)
    parser.add_argument('--lambda_2', type=float, default=1e-2)
    parser.add_argument('--lambda_3', type=float, default=1e-2)
    parser.add_argument('--K', type=int, default=500000)

    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='Checkpoints saving path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment args ' + '=' * 36)
    args = get_args()
    print(f"Using device: {args.device}")
    print(f"PyTorch Version: {torch.__version__}")
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
    train_loader, valid_loader, test_loader, in_feats, num_classes = load_dataset(args)

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    if args.dataset == 'ogbg-ppa':
        standard_model = GATGraphClassifier(feats_size=in_feats,
                                            hidden_size=128,
                                            n_classes=num_classes,
                                            n_layers=3,
                                            n_heads=[4, 2, 1],
                                            feat_drop=0.2,
                                            attn_drop=0.05,
                                            readout_type=args.readout_type).to(args.device)
        FGAI = GATGraphClassifier(feats_size=in_feats,
                                  hidden_size=128,
                                  n_classes=num_classes,
                                  n_layers=3,
                                  n_heads=[4, 2, 1],
                                  feat_drop=0.2,
                                  attn_drop=0.05,
                                  readout_type=args.readout_type).to(args.device)
        optimizer_FGAI = optim.Adam(FGAI.parameters(),
                                    lr=1e-2,
                                    weight_decay=0)
    else:
        standard_model = GATGraphClassifier(feats_size=in_feats,
                                            hidden_size=128,
                                            n_classes=num_classes,
                                            n_layers=3,
                                            n_heads=[4, 2, 1],
                                            feat_drop=0.2,
                                            attn_drop=0.05,
                                            readout_type=args.readout_type).to(args.device)
        FGAI = GATGraphClassifier(feats_size=in_feats,
                                  hidden_size=128,
                                  n_classes=num_classes,
                                  n_layers=3,
                                  n_heads=[4, 2, 1],
                                  feat_drop=0.2,
                                  attn_drop=0.05,
                                  readout_type=args.readout_type).to(args.device)
        optimizer_FGAI = optim.Adam(FGAI.parameters(),
                                    lr=1e-2,
                                    weight_decay=5e-4)
    attacker_delta = PGD(epsilon=args.epsilon,
                         n_epoch=args.n_epoch_attack,
                         n_inject_max=args.n_inject_max,
                         n_edge_max=args.n_edge_max,
                         feat_lim_min=-1,
                         feat_lim_max=1,
                         # loss=TVD,
                         device=args.device)
    attacker_rho = PGD(epsilon=args.epsilon,
                       n_epoch=args.n_epoch_attack,
                       n_inject_max=args.n_inject_max,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=-1,
                       feat_lim_max=1,
                       loss=topK_overlap_loss,
                       device=args.device)
    criterion = nn.CrossEntropyLoss()

    FGAI_trainer = FGAITrainer(FGAI, optimizer_FGAI, attacker_delta, attacker_rho, args)

    # ==================================================================================================
    # 6. Load pre-trained standard model
    # ==================================================================================================
    standard_model.load_state_dict(torch.load(f'./standard_model/{args.dataset}/model_parameters.pth'))
    standard_model.eval()

    tensor_dict = torch.load(f'./standard_model/{args.dataset}/tensors.pth')
    orig_outputs = tensor_dict['orig_outputs'].to(device=args.device)
    orig_graph_repr = tensor_dict['orig_graph_repr'].to(device=args.device)
    orig_att = tensor_dict['orig_att'].to(device=args.device)

    evaluate_graph_level(standard_model, criterion, test_loader, args.device)

    # ==================================================================================================
    # 7. Train our FGAI
    # ==================================================================================================
    FGAI_trainer.train(train_loader, valid_loader, orig_outputs, orig_graph_repr, orig_att)
    evaluate_graph_level(FGAI, criterion, test_loader, args.device)

    # ==================================================================================================
    # 7. Save FGAI
    # ==================================================================================================
    torch.save(FGAI.state_dict(), 'FGAI_parameters.pth')
