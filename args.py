import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--configs", type=str, default=None, help="configs file",
    )
    parser.add_argument(
        "--result-dir",
        default="./Results",
        type=str,
        help="directory to save results",
    )
    parser.add_argument(
        "--exp-name",
        default="alex_20nodes_4class_lr1", # inde_mask inde_bilevel
        type=str,
        help="Name of the experiment (creates dir with this name in --result-dir)",
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet18", help="Model achitecture")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes in the model",
    )

    # Pruning
    parser.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="Fraction of weight variables kept in subnet",
    )

    parser.add_argument(
        "--scale-rand-init",
        action="store_true",
        default=False,
        help="Init weight with scaling using pruning ratio",
    )

    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        default=False,
        help="freeze batch-norm parameters in pruning",
    )

    parser.add_argument(
        "--scores-init-type",
        default="kaiming_uniform",
        choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"),
        help="Which init to use for relevance scores",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet", "ImageNetOrigin", "ImageNetLMDB"],
        help="Dataset for training and eval",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=3000,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="whether to normalize the data",
    )
    parser.add_argument(
        "--data-dir", type=str, default='/mnt/HD2/yyz/Datasets/CIFAR10', help="path to datasets"
    )

    parser.add_argument(
        "--image-dim", type=int, default=32, help="Image size: dim x dim x 3"
    )

    # Training
    parser.add_argument(
        "--trainer",
        type=str,
        default="base",
        choices=["bilevel", "bilevel_finetune", "base"],
        help="Natural (base) or adversarial or verifiable training",
    )
    parser.add_argument(
        "--epochs", type=int, default=400, metavar="N", help="number of epochs to train"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=("sgd", "adam", "rmsprop")
    )
    parser.add_argument("--wd", default=5e-4, type=float, help="Weight decay")

    parser.add_argument("--mask-lr", type=float, default=10.0, help="mask learning rate for bi-level only")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--mask-lr-schedule",
        type=str,
        default="constant",
        choices=("constant", "cosine", "step"),
        help="lr scheduler for finetuning in bi-level problem"
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=("constant", "step", "cosine"),
        help="Learning rate schedule",
    )
    parser.add_argument("--momentum", type=float, default=0.0, help="SGD momentum")

    # Additional
    parser.add_argument("--seed", type=int, default=1234, help="random seed")

    parser.add_argument(
        "--lr2",
        type=float,
        default=0.1,
        help="learning rate for the second term",
    )

    # the distributed parameters
    parser.add_argument(
        "--num_clients",
        type=int,
        default=20
    )

    parser.add_argument(
        "--num_localeps",
        type=int,
        default=1
    )

    return parser.parse_args()
