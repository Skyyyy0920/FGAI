import time
import argparse
from model import *
from dataset import *
from trainer import *
from attacker import *
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    # setup_seed(args.seed)  # make the experiment repeatable
    print('\n' + '=' * 36 + ' Setup logger ' + '=' * 36)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join("standard_model", f"{args.dataset}")
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

    g, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    features = g.ndata["feat"]
    num_feats = features.shape[1]

    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'ogbn-arxiv':
        standard_model = GATNodeClassifier(in_feats=num_feats, hid_dim=256, n_classes=num_classes, n_layers=3,
                                           n_heads=[4, 2, 2, 1], feat_drop=0.4, attn_drop=0.05).to(args.device)
        optimizer = optim.Adam(standard_model.parameters(), lr=5e-3, weight_decay=0)
    else:
        standard_model = GATNodeClassifier(in_feats=num_feats, hid_dim=8, n_classes=num_classes, n_layers=1,
                                           n_heads=[8, 1]).to(args.device)
        optimizer = optim.Adam(standard_model.parameters(), lr=1e-3, weight_decay=5e-4)

    total_params = sum(p.numel() for p in standard_model.parameters())
    print(f"Total parameters: {total_params}")

    std_trainer = StandardTrainer(standard_model, criterion, optimizer, args)

    orig_outputs, orig_graph_repr, orig_att = std_trainer.train(g, features, label, train_idx, valid_idx)

    evaluate(standard_model, criterion, g, features, label, test_idx)

    torch.save(standard_model.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))
    tensor_dict = {'orig_outputs': orig_outputs, 'orig_graph_repr': orig_graph_repr, 'orig_att': orig_att}
    torch.save(tensor_dict, os.path.join(save_dir, 'tensors.pth'))
