from itertools import chain, repeat, islice
from collections import defaultdict

import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

def in_jupyter():
    try:
        from IPython import get_ipython
        from ipykernel.zmqshell import ZMQInteractiveShell
        assert isinstance(get_ipython(), ZMQInteractiveShell)
    except Exception:
        return False
    return True

try:
    if in_jupyter():
        # from tqdm import tqdm as pbar
        from tqdm import tqdm_notebook as pbar
    else:
        from tqdm import tqdm as pbar
except ImportError:
    def pbar(it, *a, **kw):
        return it
    
# avoid log(0)
_eps = 1e-15

def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False
def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def _as_tuple(x,n):
    if not isinstance(x, tuple):
        return (x,)*n
    assert len(x)==n, 'input is a tuple of incorrect size'
    return x
            
def _take_epochs(X, n_epochs):
    """Get a fractional number of epochs from X, rounded to the batch
    X: torch.utils.DataLoader (has len(), iterates over batches)
    n_epochs: number of iterations through the data.
    """
    n_batches = int(np.ceil(len(X)*n_epochs))
    _take_iters(X, n_batches)

def _take_batches(X, n_batches):
    """Get a integer number of batches from X, reshuffling as necessary
    X: torch.utils.DataLoader (has len(), iterates over batches)
    n_batches: number of batches
    """
    n_shuffles = int(np.ceil(len(X)/n_batches))
    return islice(chain.from_iterable(repeat(X,n_shuffles)),n_batches)

                
class BiGAN(nn.Module):
    def __init__(self, E, G, D, latent_dim, x_cycle_weight=0, z_cycle_weight=0):
        """
        E: nn.Module mapping X to Z
        G: nn.Module mapping Z to X
        D: nn.module discriminating (real x, inferred z) from (generated x, sampled z)
        latent_dim: dimensionality of Z
        """
        super().__init__()
        self.E = E
        self.G = G
        self.D = D
        self.latent_dim = latent_dim
        self.x_cycle_weight = x_cycle_weight
        self.z_cycle_weight = z_cycle_weight

    def is_cuda(self):
        return any(p.is_cuda for p in self.parameters())

    def _wrap(self, x, **kwargs):
        """ensure x is a Variable on the correct device"""
        if not isinstance(x, Variable):
            # if x isn't a Tensor, attempt to construct one from it
            if not isinstance(x, torch._TensorBase):
                x = torch.Tensor(x)
            x = Variable(x, **kwargs)
        if self.is_cuda():
            x = x.cuda()
        return x
    
    def sample_prior(self, n):
        """Sample self.latent_dim-dimensional unit normal.
        n: batch size
        """
#         return self._wrap(torch.randn(n, self.latent_dim))
#         return self._wrap(torch.rand(n, self.latent_dim)*2-1)
        z = torch.randn(n, self.latent_dim)
        z = z/(z*z).sum(1).sqrt().unsqueeze(1)
        return self._wrap(z)

    def objective(self, x):
        """Objective to minimize for D, E/G"""
        
        z = self.sample_prior(len(x))
        x_gen = self.G(z)
        z_enc = self.E(x)
        xs = torch.cat((x, x_gen))
        zs = torch.cat((z_enc, z))
        d_real, d_fake = self.D(xs, zs).chunk(2)
        
        cycle_loss = 0
        if self.x_cycle_weight > 0:
            x_hat = self.G(z_enc)
            cycle_loss += self.x_cycle_weight * (x-x_hat).abs().mean()
        if self.z_cycle_weight > 0:
            z_hat = self.E(x_gen)
            cycle_loss += self.z_cycle_weight * (z-z_hat).abs().mean()
            
        return {
            'EG_loss': 
                cycle_loss
                - (1 - d_real + _eps).log().mean()
                - (d_fake + _eps).log().mean(),
            'D_loss':
                - (1 - d_fake + _eps).log().mean()
                - (d_real + _eps).log().mean()
        }
    
    def _epoch(self, X, D_opt=None, EG_opt=None, n_batches=None):
        """Evaluate/optimize for one epoch.
        X: torch.nn.DataLoader
        D_opt, EG_opt: torch.nn.Optimizer or None if not training
        n_batches: number of batches to draw or None for all data
        """
        iter_objs = defaultdict(list)
        
        training = bool(D_opt and EG_opt)
        it = _take_batches(X, n_batches) if n_batches else X
        desc = 'training batch' if training else 'validating batch'
        for x in pbar(it, desc=desc, leave=False):
            x = self._wrap(x)
            obj = self.objective(x)
            if training:
                self.zero_grad()
                obj['D_loss'].backward(retain_graph=True)
                D_opt.step()
                self.zero_grad()
                obj['EG_loss'].backward()
                EG_opt.step()
            for k,v in obj.items():
                iter_objs[k].append(v.data.cpu().numpy())
            del obj
        return {k:np.mean(v) for k,v in iter_objs.items()}
    
    def fit(self,
            X_train, X_valid=None,
            opt_fn=torch.optim.Adam, opt_params={'lr':2e-4, 'betas':(.5,.999)},
            n_batches=None, n_epochs=10,
            log_fn=None, log_every=1,
            checkpoint_fn=None, checkpoint_every=2):
        """
        X_train: torch.utils.data.DataLoader
        X_valid: torch.utils.data.DataLoader or None
        opt_fn: nn.Optimizer constructor or triple for D, E/G
        opt_params: dict of keyword args for optimizer or triple for D, E/G
        n_batches: int or pair # of train, valid batches per epoch (None for all data)
        n_epochs: number of training iterations
        log_fn: takes diagnostic dict, called after every nth epoch
        log_every: call log function every nth epoch
        checkpoint_fn: takes model, epoch. called after nth every epoch
        checkpoint_every: call checkpoint function every nth epoch
        """       
        _unfreeze(self)
        
        train_batches, valid_batches = _as_tuple(n_batches, 2)
        D_opt_fn, EG_opt_fn = (
            lambda p: fn(p, **hyperparams) for fn, hyperparams in zip(
                _as_tuple(opt_fn, 2),  _as_tuple(opt_params, 2)))
        
        EG_opt = EG_opt_fn(chain( self.E.parameters(), self.G.parameters()))
        D_opt = D_opt_fn(self.D.parameters())
        #LambdaLR(D_opt, lambda _:-1) # negate learning rate for D_opt
        
        for i in pbar(range(n_epochs), desc='epoch'):
            diagnostic = defaultdict(dict)
            report = log_fn and (i%log_every==0 or i==n_epochs-1)
            checkpoint = checkpoint_every and checkpoint_fn and (
                (i+1)%checkpoint_every==0 or i==n_epochs-1 )
            # train for one epoch
            self.train()
            _unfreeze(self)
            diagnostic['train'].update( self._epoch(
                X_train, D_opt, EG_opt, n_batches=train_batches ))
            # validate for one epoch
            self.eval()
            _freeze(self)
            diagnostic['valid'].update(self._epoch(
                X_valid, n_batches=valid_batches ))
            # log the dict of loss components
            if report:
                log_fn(diagnostic)
            if checkpoint:
                checkpoint_fn(self, i+1)

    def forward(self, *args, mode=None):
        """
        mode:
            None: return z = E(x), x_rec = G(z); args[0] is x.
            sample: return z = P(z), x = G(z); args[0] is number of samples.
            generate: return x = G(z); args[0] is z.
            encode: return z = E(x); args[0] is x.
            reconstruct: like None, but only return x_rec.
        """
        # get code from prior, args, or by encoding
        if mode=='sample':
            n = args[0]
            z = self.sample_prior(n)
        elif mode=='generate':
            z = self._wrap(args[0])
        else:
            x = self._wrap(args[0])
            z = self.E(x)
        # step there if reconstruction not desired
        if mode=='encode':
            return z
        # run code through G
        x_rec = self.G(z)
        if mode=='reconstruct' or mode=='generate':
            return x_rec
        # default, 'sample': return code and reconstruction
        return z, x_rec
    
class BiLSGAN(BiGAN):
    def objective(self, x):
        """Objective to minimize for D, E/G"""

        z = self.sample_prior(len(x))
        
        xs = torch.cat((x, self.G(z)))
        zs = torch.cat((self.E(x), z))
        ds = self.D(xs, zs)
        d_real, d_fake = ds.chunk(2)
        
        return {
            'EG_loss':
                (ds*ds).mean(),
            'D_loss':
                ((d_real-1)**2 + (d_fake+1)**2).mean()
        }
#         return {
#             'EG_loss':
#                 ((d_fake-1)*(d_fake-1) + d_real*d_real).mean(),
#             'D_loss':
#                 ((d_real-1)*(d_real-1) + d_fake*d_fake).mean()
#         }

class BiWGAN(BiGAN):
    def objective(self, x):
        """Objective to minimize for D, E/G"""
        z = self.sample_prior(len(x))
        z_enc = self.E(x)
        x_gen = self.G(z)
        
        xs = torch.cat((x, x_gen))
        zs = torch.cat((z_enc, z))
        ds = self.D(xs, zs)
        d_real, d_fake = ds.chunk(2)
        
        return {
            'EG_loss':
                (d_real-d_fake).mean(),
            'D_loss':
                (d_fake-d_real).mean() + self.gradient_penalty(x, x_gen, z, z_enc)
        }
    
    def gradient_penalty(self, x, x_gen, z, z_enc, w=10):
        """WGAN-GP gradient penalty"""
        assert x.size()==x_gen.size(), 'real and sampled x sizes do not match'
        assert z.size()==z_enc.size(), 'encoded and sampled z sizes do not match'
        assert len(x)==len(z), 'x and z batch sizes do not match'
        batch_size = len(x)
        alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
        alpha = alpha_t(batch_size).uniform_()
        z_hat = z_enc.data*alpha + z.data*(1-alpha)
        z_hat = Variable(z_hat, requires_grad=True)
        alpha = alpha.view(batch_size, 1, 1, 1)
        x_hat = x.data*alpha + x_gen.data*(1-alpha)
        x_hat = Variable(x_hat, requires_grad=True)

        def eps_norm(x):
            return (x*x+_eps).sum(-1).sqrt()
        def bi_penalty(x):
            return (x-1)**2

        d = self.D(x_hat, z_hat).sum()
        grad_xhat = torch.autograd.grad(
            d, x_hat, create_graph=True, only_inputs=True
        )[0]
        grad_zhat = torch.autograd.grad(
            d, z_hat, create_graph=True, only_inputs=True
        )[0]
    
        grads = torch.cat((grad_xhat.flatten(), grad_zhat.flatten()))
        penalty = w*bi_penalty(eps_norm(grads)).mean()
        return penalty