import time
import argparse
import torch.nn as nn
import torch.optim as optim
from utils import *
from models import GATGraphClassifier
from load_dataset import load_dataset
from trainer import StandardTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()

    # Operation environment
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=device, help='Running on which device')

    # Data
    parser.add_argument('--dataset', type=str,
                        # default='ogbg-ppa',
                        # default='ogbg-molhiv',
                        # default='MUTAG',
                        default='PROTEINS',
                        # default='IMDBBINARY',
                        # default='IMDBMULTI',
                        help='Dataset name')

    # Experimental Setup
    parser.add_argument('--num_epochs', type=int, default=200, help='Training epoch')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--readout_type', type=str,
                        # default='K-shell',
                        default='mean',
                        # default='max',
                        # default='min'
                        help='Readout graph')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(f"Using device: {args.device}")
    print(f"PyTorch Version: {torch.__version__}")
    current_dir = os.getcwd()
    print("Current work dir：", current_dir)
    new_dir = current_dir + "/Node Classification"
    os.chdir(new_dir)
    print('\n' + '=' * 36 + ' Setup logger ' + '=' * 36)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("checkpoints", f"{args.base_model}+vanilla", f"{args.dataset}_{logging_time}")
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

    train_loader, valid_loader, test_loader, in_feats, num_classes = load_dataset(args)

    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'ogbg-ppa':
        standard_model = GATGraphClassifier(feats_size=in_feats,
                                            hidden_size=128,
                                            n_classes=num_classes,
                                            n_layers=3,
                                            n_heads=[4, 2, 1],
                                            feat_drop=0.2,
                                            attn_drop=0.05,
                                            readout_type=args.readout_type).to(args.device)
        optimizer = optim.Adam(standard_model.parameters(),
                               lr=1e-4,
                               weight_decay=0)
    else:
        standard_model = GATGraphClassifier(feats_size=in_feats,
                                            hidden_size=64,
                                            n_classes=num_classes,
                                            n_layers=3,
                                            n_heads=[4, 2, 1],
                                            feat_drop=0.2,
                                            attn_drop=0.2,
                                            readout_type=args.readout_type).to(args.device)
        optimizer = optim.Adam(standard_model.parameters(),
                               lr=1e-2,
                               weight_decay=5e-4)

    total_params = sum(p.numel() for p in standard_model.parameters())
    logging.info(f"Total parameters: {total_params}")

    std_trainer = StandardTrainer(standard_model, criterion, optimizer, args)

    orig_outputs, orig_graph_repr, orig_att = std_trainer.train(train_loader, valid_loader)

    evaluate_graph_level(standard_model, test_loader, args.device)

    torch.save(standard_model.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))
    tensor_dict = {'orig_outputs': orig_outputs, 'orig_graph_repr': orig_graph_repr, 'orig_att': orig_att}
    torch.save(tensor_dict, os.path.join(save_dir, 'tensors.pth'))
