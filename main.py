import yaml
import time
import pickle
import zipfile
import logging
from model import *
from utils import *
from config import *
from dataset import *

if __name__ == '__main__':
    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment args ' + '=' * 36)
    args = get_args()
    print(f"Using device: {args.device}")
    setup_seed(args.seed)  # make the experiment repeatable

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
    num_classes = dataset.num_classes
    g = dataset[0].to(device=args.device)
    num_feats = g.ndata["feat"].shape[1]
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    ss = g.ndata['feat']
    train_feat, train_label = g.ndata['feat'][train_mask], g.ndata['label'][train_mask]
    val_feat, val_label = g.ndata['feat'][val_mask], g.ndata['label'][val_mask]
    test_feat, test_label = g.ndata['feat'][test_mask], g.ndata['label'][test_mask]

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================

    # ==================================================================================================
    # 6. Load pre-trained model
    # ==================================================================================================

    # ==================================================================================================
    # 7. Training
    # ==================================================================================================

    # ==================================================================================================
    # 8. Validation and Testing
    # ==================================================================================================
