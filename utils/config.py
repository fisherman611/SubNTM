import argparse


def new_parser(name=None):
    return argparse.ArgumentParser(prog=name)


def add_dataset_argument(parser):
    parser.add_argument('--dataset', type=str,
                        help='dataset name', default='20NG')
    
def add_logging_argument(parser):
    parser.add_argument('--wandb_prj', type=str, default='SubNTM')


def add_model_argument(parser):
    parser.add_argument('--en1_units', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--adapter_alpha', type=float, default=0.1)
    parser.add_argument('--beta_temp', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--weight_loss_ECR', type=float, default=200.0)
    parser.add_argument('--sinkhorn_alpha', type=float, default=20.0)
    parser.add_argument('--sinkhorn_max_iter', type=int, default=1000)
    parser.add_argument('--augment_coef', type=float, default=0.0)
    parser.add_argument('--lambda_doc', type=float, default=1.0)


def add_training_argument(parser):
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the model, cuda or cpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_scheduler', type=str,
                        help='learning rate scheduler, dont use if not needed, \
                            currently support: step', default='StepLR')
    parser.add_argument('--lr_step_size', type=int, default=125,
                        help='step size for learning rate scheduler')

def add_eval_argument(parser):
    parser.add_argument('--tune_SVM', action='store_true', default=False)


def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_config(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if value.isdigit():
                if value.find('.') != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args
