import json
import argparse
from trainer import train
import wandb

def main():
    args = setup_parser().parse_args()
    args.config = f"./exps/{args.model_name}.json"
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)

    wandb.init(name=args['logpath'], project=f'memcil-{args["dataset"]}')
    wandb.config.update(args)

    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')

    parser.add_argument('--dataset', type=str, default="cifar100")
    parser.add_argument('--memory_size','-ms',type=int, default=2000)
    parser.add_argument('--init_cls', '-init', type=int, default=10)
    parser.add_argument('--increment', '-incre', type=int, default=10)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--ntokens', type=int, default=10)
    parser.add_argument('--model_name','-model', type=str, default=None, required=True)
    parser.add_argument('--convnet_type','-net', type=str, default='resnet32')
    parser.add_argument('--prefix','-p',type=str, help='exp type', default='benchmark', choices=['benchmark', 'fair', 'auc'])
    parser.add_argument('--ver', type=str, help='memknn model type', default='m8', choices=['m8', 'm8_4', 'm18', 'nakata', 'p19_1'])
    parser.add_argument('--device','-d', nargs='+', type=int, default=[0,1,2,3])
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--skip', action="store_true")
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--distillation', action="store_true")
    parser.add_argument('--normalize', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--logpath', type=str, default='test')
    parser.add_argument('--base_scale', type=float, default=8.)
    parser.add_argument('--knn_scale', type=float, default=32.)
    parser.add_argument('--epoch', type=int, default=1)

    return parser


if __name__ == '__main__':
    main()
