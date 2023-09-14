import time
from model import *
from config import *
from dataset import *
from trainer import *
from attacker import *
import torch.optim as optim

if __name__ == '__main__':
    args = get_args()
    print('\n' + '=' * 36 + ' Setup logger ' + '=' * 36)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"standard_model_{args.dataset}_{logging_time}")
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

    dataset = load_dataset(args)
    g = dataset[0].to(device=args.device)
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    train_label = g.ndata['label'][train_mask]
    val_label = g.ndata['label'][val_mask]
    test_label = g.ndata['label'][test_mask]
    features = g.ndata["feat"]
    num_feats = features.shape[1]
    num_classes = dataset.num_classes

    criterion = nn.CrossEntropyLoss()
    standard_model = GATNodeClassifier(in_feats=num_feats, hid_dim=8, n_classes=num_classes, n_layers=1,
                                       n_heads=[8, 1]).to(args.device)
    optimizer = optim.Adam(standard_model.parameters(), lr=5e-3, weight_decay=5e-4)

    std_trainer = StandardTrainer(standard_model, criterion, optimizer, args)

    m_l = train_mask, train_label, val_mask, val_label
    orig_outputs, orig_graph_repr, orig_att = std_trainer.train(g, features, m_l)

    evaluate(standard_model, criterion, g, features, test_mask, test_label)

    torch.save(standard_model.state_dict(), 'model_parameters.pth')
    tensor_dict = {'orig_outputs': orig_outputs, 'orig_graph_repr': orig_graph_repr, 'orig_att': orig_att}
    torch.save(tensor_dict, 'tensors.pth')
