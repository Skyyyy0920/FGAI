import yaml
import time
import pickle
import zipfile
from model import *
from config import *
from dataset import *
from trainer import *
from attacker import *
import torch.optim as optim

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
    # src, dst = g.edges()
    # print(src, dst)

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    model = GATNodeClassifier(in_feats=num_feats, hid_dim=8, n_classes=num_classes,
                              n_layers=1, n_heads=[8, 1]).to(device=args.device)
    PGDer = PGDAttacker(radius=args.pgd_radius, steps=args.pgd_step, step_size=args.pgd_step_size,
                        random_start=True, norm_type=args.pgd_norm_type, ascending=True)
    X_PGDer = PGDAttacker(radius=args.x_pgd_radius, steps=args.x_pgd_step, step_size=args.x_pgd_step_size,
                          random_start=True, norm_type=args.x_pgd_norm_type, ascending=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    trainer = Trainer(model, criterion, optimizer, PGDer, X_PGDer, args)

    # ==================================================================================================
    # 6. Load pre-trained model
    # ==================================================================================================

    # ==================================================================================================
    # 7. Training and Validation
    # ==================================================================================================
    trainer.train(g, features, train_mask, train_label, val_mask, val_label)

    # ==================================================================================================
    # 8. Testing
    # ==================================================================================================
    trainer.evaluate(g, features, test_mask, test_label)
