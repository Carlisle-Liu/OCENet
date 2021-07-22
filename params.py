import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--epoch', type=int, default=40, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--lr_dis', type=float, default=1e-5, help='learning rate')
parser.add_argument('--lr_des', type=float, default=2.5e-5, help='learning rate for descriptor')
parser.add_argument('--batchsize', type=int, default=5, help='training batch size')
parser.add_argument('--batchsize_test', type=int, default=1, help='test batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('--beta_gen', type=float, default=0.5, help='beta of Adam for generator')
parser.add_argument('--beta_dis', type=float, default=0.5, help='beta of Adam for descriptor')
parser.add_argument('--gen_reduced_channel', type=int, default=32, help='reduced channel in generator')
parser.add_argument('--des_reduced_channel', type=int, default=64, help='reduced channel in descriptor')
parser.add_argument('--langevin_step_num_des', type=int, default=10, help='number of langevin steps for ebm')
parser.add_argument('--langevin_step_size_des', type=float, default=0.026, help='step size of EBM langevin')
parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')
parser.add_argument('--train_image_root', default='/students/u6617221/COD-master/dataset/train/Imgs/',
                    help='training image root directory')
parser.add_argument('--train_gt_root', default='/students/u6617221/COD-master/dataset/train/GT/',
                    help='training ground truth root directory')
parser.add_argument('--train_bd_root', default='/students/u6617221/COD-master/dataset/train/Edge/',
                    help='training boundary label root directory')
parser.add_argument('--test_dataset_root', default='/students/u6617221/COD-master/dataset/test/',
                    help='testing image root directory')
parser.add_argument('--model_save_path', default='./checkpoint/', help='directory where training model is saved to')
parser.add_argument('--experiment_name', default='S1', help='name of the experiment where model checkpoints are saved to')
parser.add_argument('--test_datasets', default=['CAMO', 'CHAMELEON', 'COD10K'],
                    help='collection of testing dataset')
parser.add_argument('--eval_save_path', default='/students/u6617221/COD-master/train_val/')
parser.add_argument('--seed', type=int, default=1, help='random seed')
