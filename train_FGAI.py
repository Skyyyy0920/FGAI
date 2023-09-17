import yaml
import time
import zipfile
import argparse
from model import *
from dataset import *
from trainer import *
from attacker import *
import torch.optim as optim

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

    parser.add_argument('--pgd_radius', type=float, default=0.05, help='Attack radius')
    parser.add_argument('--pgd_step', type=float, default=10, help='How many step to conduct PGD')
    parser.add_argument('--pgd_step_size', type=float, default=0.02, help='Coefficient of PGD')
    parser.add_argument('--pgd_norm_type', type=str, default="l-infty", help='Which norm of your noise')

    parser.add_argument('--lambda_1', type=float, default=5e-2)
    parser.add_argument('--lambda_2', type=float, default=5e-2)
    parser.add_argument('--lambda_3', type=float, default=5e-2)
    parser.add_argument('--K', type=int, default=4)

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
    g, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    features = g.ndata["feat"]
    num_feats = features.shape[1]

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    standard_model = GATNodeClassifier(in_feats=num_feats, hid_dim=8, n_classes=num_classes,
                                       n_layers=1, n_heads=[8, 1]).to(device=args.device)
    FGAI = GATNodeClassifier(in_feats=num_feats, hid_dim=8, n_classes=num_classes,
                             n_layers=1, n_heads=[8, 1]).to(device=args.device)
    PGDer = PGDAttacker(radius=args.pgd_radius, steps=args.pgd_step, step_size=args.pgd_step_size,
                        random_start=True, norm_type=args.pgd_norm_type, ascending=True, device=args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer_FGAI = optim.Adam(FGAI.parameters(), lr=1e-2, weight_decay=5e-4)

    FGAI_trainer = FGAITrainer(FGAI, optimizer_FGAI, PGDer, args)

    # ==================================================================================================
    # 6. Load pre-trained standard model
    # ==================================================================================================
    standard_model.load_state_dict(torch.load(f'./standard_model/{args.dataset}/model_parameters.pth'))
    standard_model.eval()

    tensor_dict = torch.load(f'./standard_model/{args.dataset}/tensors.pth')
    orig_outputs = tensor_dict['orig_outputs'].to(device=args.device)
    orig_graph_repr = tensor_dict['orig_graph_repr'].to(device=args.device)
    orig_att = tensor_dict['orig_att'].to(device=args.device)

    evaluate(standard_model, criterion, g, features, label, test_idx)

    # ==================================================================================================
    # 7. Train our FGAI
    # ==================================================================================================
    FGAI_trainer.train(g, features, label, train_idx, valid_idx, orig_outputs, orig_graph_repr, orig_att, criterion)
    evaluate(FGAI, criterion, g, features, label, test_idx)

    # ==================================================================================================
    # 7. Save FGAI
    # ==================================================================================================
    torch.save(FGAI.state_dict(), 'FGAI_parameters.pth')
