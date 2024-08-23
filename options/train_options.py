from .base_options import BaseOptions

class TrainOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--min_lr', type=float, default=1e-6, help='initial learning rate for adam')
        self.parser.add_argument('--update_learning_rate', type=int, default=0, help='whether to update learning rate')
        self.parser.add_argument('--warmup_epochs', type=float, default=0, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')


        # display stuff
        self.parser.add_argument('--display_freq', type=int, default=3000, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=25, help='frequency of showing training results on console')
        self.parser.add_argument('--ckpt_num', type=int, default=5, help='The number of checkpoint kept')

        self.parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_steps_freq', type=int, default=1000, help='frequency of saving checkpoints')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--ema_rate', type=float, default=0.999, help='the rate of Exponential Moving Average')

        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--total_iters', type=int, default=100000000, help='# of iter for training')
        self.parser.add_argument('--epochs', type=int, default=4000, help='# of iter for training')
        self.parser.add_argument('--start_iter', type=int, default=0, help='# of iter for training')

        self.parser.add_argument('--mode', type=str, default='train', help='# of iter for training', choices=["train", "generate"])
        self.parser.add_argument('--isTrain', type=str, default='True', help='# of iter for training')
