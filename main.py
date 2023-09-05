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
    # train_feat, train_label = g.ndata['feat'][train_mask], g.ndata['label'][train_mask]
    # val_feat, val_label = g.ndata['feat'][val_mask], g.ndata['label'][val_mask]
    # test_feat, test_label = g.ndata['feat'][test_mask], g.ndata['label'][test_mask]
    test_feat, test_label = g.ndata['feat'][train_mask], g.ndata['label'][train_mask]
    val_feat, val_label = g.ndata['feat'][val_mask], g.ndata['label'][val_mask]
    train_feat, train_label = g.ndata['feat'][test_mask], g.ndata['label'][test_mask]

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================

    # Initialize the model
    model = GATNodeClassifier(in_feats=num_feats, hidden_dim=64, n_classes=num_classes, dropout=0.5, alpha=0.2,
                              n_heads=4).to(device=args.device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        # outputs = model(train_feat, g.subgraph(train_mask).adjacency_matrix().to_dense())
        outputs = model(train_feat, g.subgraph(test_mask).adjacency_matrix().to_dense())
        loss = criterion(outputs, train_label)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_feat, g.subgraph(val_mask).adjacency_matrix().to_dense())
            val_loss = criterion(val_outputs, val_label)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(val_label.cpu(), val_preds.cpu())

        print(
            f'Epoch [{epoch + 1}/{args.num_epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')

    # Testing
    model.eval()
    with torch.no_grad():
        # test_outputs = model(test_feat, g.subgraph(test_mask).adjacency_matrix().to_dense())
        test_outputs = model(test_feat, g.subgraph(train_mask).adjacency_matrix().to_dense())
        test_loss = criterion(test_outputs, test_label)
        test_preds = torch.argmax(test_outputs, dim=1)
        test_accuracy = accuracy_score(test_label.cpu(), test_preds.cpu())

    print(f'Test Loss: {test_loss.item():.4f} | Test Accuracy: {test_accuracy:.4f}')

    # ==================================================================================================
    # 6. Load pre-trained model
    # ==================================================================================================

    # ==================================================================================================
    # 7. Training
    # ==================================================================================================

    # ==================================================================================================
    # 8. Validation and Testing
    # ==================================================================================================
