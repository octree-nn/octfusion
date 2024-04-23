from termcolor import colored
import torch

from models.networks.vqvae_networks.network import VQVAE
from models.networks.dualoctree_networks.graph_ae import GraphAE
# from models.networks.dualoctree_networks.graph_vqvae import GraphVQVAE
from models.networks.dualoctree_networks.graph_vqvae_v1 import GraphVQVAE
from models.networks.dualoctree_networks.graph_vae import GraphVAE

def load_dualoctree(conf, ckpt, opt = None):
    assert type(ckpt) == str
    flags = conf.model
    params = [flags.depth, flags.channel, flags.nout,
            flags.full_depth, flags.depth_stop, flags.depth_out, flags.use_checkpoint]
    if flags.name == 'graph_ounet' or \
        flags.name == 'graph_unet' or \
        flags.name == 'graph_ae' or \
        flags.name == 'graph_vqvae_v0' or \
        flags.name == 'graph_vqvae_v1' or \
        flags.name == 'graph_vae':
        params.append(flags.resblock_type)
        params.append(flags.bottleneck)
        params.append(flags.resblk_num)

    if flags.name == 'graph_ae':
        params.append(flags.code_channel)
        dualoctree = GraphAE(*params)

    if flags.name == 'graph_vqvae' or \
        flags.name == 'graph_vqvae_v1':
        params.append(flags.code_channel)
        params.append(flags.embed_dim)
        params.append(flags.n_embed)
        dualoctree = GraphVQVAE(*params)

    if flags.name == 'graph_vae':
        params.append(flags.code_channel)
        params.append(flags.embed_dim)
        dualoctree = GraphVAE(*params)

    trained_dict = torch.load(ckpt, map_location='cuda')
    if ckpt.endswith('.solver.tar'):
        model_dict = trained_dict['model_dict']
    else:
        model_dict = trained_dict

    # dualocnn = dualocnn.module if opt.distributed else dualocnn
    dualoctree.load_state_dict(model_dict)

    print(colored('[*] DualOctree: weight successfully load from: %s' % ckpt, 'blue'))
    dualoctree.requires_grad = False

    dualoctree.to(opt.device)
    dualoctree.eval()
    return dualoctree


def load_vqvae(vq_conf, vq_ckpt, opt=None):
    assert type(vq_ckpt) == str

    # init vqvae for decoding shapes
    mparam = vq_conf.model.params
    n_embed = mparam.n_embed
    embed_dim = mparam.embed_dim
    ddconfig = mparam.ddconfig

    n_down = len(ddconfig.ch_mult) - 1

    vqvae = VQVAE(ddconfig, n_embed, embed_dim)

    map_fn = lambda storage, loc: storage
    state_dict = torch.load(vq_ckpt, map_location=map_fn)
    if 'vqvae' in state_dict:
        vqvae.load_state_dict(state_dict['vqvae'])
    else:
        vqvae.load_state_dict(state_dict)

    print(colored('[*] VQVAE: weight successfully load from: %s' % vq_ckpt, 'blue'))
    vqvae.requires_grad = False

    vqvae.to(opt.device)
    vqvae.eval()
    return vqvae
