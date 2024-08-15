from termcolor import colored
import torch

from models.networks.dualoctree_networks.graph_vae import GraphVAE

def load_dualoctree(conf, ckpt, opt = None):
    flags = conf.model
    params = [flags.depth, flags.channel, flags.nout,
            flags.full_depth, flags.depth_stop, flags.depth_out, flags.use_checkpoint]
    if flags.name == 'graph_vae':
        params.append(flags.resblock_type)
        params.append(flags.bottleneck)
        params.append(flags.resblk_num)
        params.append(flags.code_channel)
        params.append(flags.embed_dim)
        dualoctree = GraphVAE(*params)

    if ckpt is not None:
        trained_dict = torch.load(ckpt, map_location='cuda')
        if ckpt.endswith('.solver.tar'):
            model_dict = trained_dict['model_dict']
        else:
            model_dict = trained_dict
        
        if 'autoencoder' in model_dict:
            model_dict = model_dict['autoencoder']

        dualoctree.load_state_dict(model_dict)

    print(colored('[*] DualOctree: weight successfully load from: %s' % ckpt, 'blue'))
    dualoctree.requires_grad = False

    dualoctree.to(opt.device)
    dualoctree.eval()
    return dualoctree