import yaml
import time
import pickle
import zipfile
import logging
from model import *
from utils import *
from config import *
from dataset import *
import torch.optim as optim
from sklearn.metrics import accuracy_score

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
    num_classes = dataset.num_classes
    g = dataset[0].to(device=args.device)
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    train_label = g.ndata['label'][train_mask]
    val_label = g.ndata['label'][val_mask]
    test_label = g.ndata['label'][test_mask]
    features = g.ndata["feat"]
    num_feats = features.shape[1]

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================

    # Initialize the model
    model = GATNodeClassifier(in_feats=num_feats, hid_dim=8, num_classes=num_classes, num_layers=1,
                              num_heads=[8, 1]).to(device=args.device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    # ==================================================================================================
    # 6. Load pre-trained model
    # ==================================================================================================

    # ==================================================================================================
    # 7. Training
    # ==================================================================================================
    for epoch in range(args.num_epochs):
        model.train()

        # Forward pass
        # outputs = model(features, g.adjacency_matrix().to_dense())
        outputs = model(g, features)
        loss = criterion(outputs[train_mask], train_label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            # val_outputs = model(features, g.adjacency_matrix().to_dense())
            val_outputs = model(g, features)
            val_loss = criterion(val_outputs[val_mask], val_label)
            val_preds = torch.argmax(val_outputs[val_mask], dim=1)
            val_accuracy = accuracy_score(val_label.cpu(), val_preds.cpu())

        logging.info(
            f'Epoch [{epoch + 1}/{args.num_epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')

    # ==================================================================================================
    # 8. Validation and Testing
    # ==================================================================================================
    model.eval()
    with torch.no_grad():
        # test_outputs = model(features, g.adjacency_matrix().to_dense())
        test_outputs = model(g, features)
        test_loss = criterion(test_outputs[test_mask], test_label)
        test_preds = torch.argmax(test_outputs[test_mask], dim=1)
        test_accuracy = accuracy_score(test_label.cpu(), test_preds.cpu())

    logging.info(f'Test Loss: {test_loss.item():.4f} | Test Accuracy: {test_accuracy:.4f}')
