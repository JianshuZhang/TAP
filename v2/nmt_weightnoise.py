'''
Build a TAP model with guided hybrid attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
#import ipdb
import numpy
import copy
import pprint
import math
import os
import warnings
import sys
import time

from collections import OrderedDict

from data_iterator import dataIterator,dataIterator_valid
from optimizers import adadelta, adadelta_weightnoise

profile = False

import random
import re

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def conv_norm_weight(kernel_size, nin, nout=None, scale=0.01):
    W = scale * numpy.random.rand(nout, nin, kernel_size, 1)
    return W.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(options, seqs_x, seqs_y, seqs_a, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_seqs_a = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y, s_a in zip(lengths_x, seqs_x, lengths_y, seqs_y, seqs_a):
            if l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_seqs_a.append(s_a)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        seqs_a = new_seqs_a

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples, options['dim_feature'])).astype('float32') # SeqX * batch * dim
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64') # the <eol> must be 0 in the dict !!!
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    a = numpy.zeros((maxlen_x, n_samples, maxlen_y)).astype('float32') # SeqX * batch * SeqY
    a_mask = numpy.zeros((maxlen_x, n_samples, maxlen_y)).astype('float32') # SeqX * batch * SeqY
    for idx, [s_x, s_y, s_a] in enumerate(zip(seqs_x, seqs_y, seqs_a)):
        x[:lengths_x[idx], idx,:] = s_x # the zeros frame is a padding frame to align <eol>
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.
        a[:lengths_x[idx], idx,:lengths_y[idx]] = s_a * 1.
        a_mask[:lengths_x[idx]+1, idx,:lengths_y[idx]+1] = 1.

    return x, x_mask, y, y_mask, a, a_mask


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None, dimatt=None, 
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if dimatt is None:
        dimatt = options['dim_attention']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to gru
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimatt)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, dimatt)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimatt,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimatt, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    # coverage conv
    params[_p(prefix, 'conv_Q')] = conv_norm_weight(options['kernel_coverage'], 1, options['dim_coverage']).astype('float32')
    params[_p(prefix, 'conv_Uf')] = norm_weight(options['dim_coverage'], dimatt)
    params[_p(prefix, 'conv_b')] = numpy.zeros((dimatt,)).astype('float32')
    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None, alpha_past=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]
    dimctx = tparams[_p(prefix, 'Wcx')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    if alpha_past is None:
        alpha_past = tensor.alloc(0., n_samples, context.shape[0])

    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, alpha_past_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl, conv_Q, conv_Uf, conv_b):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)

        # converage vector
        cover_F = theano.tensor.nnet.conv2d(alpha_past_[:,None,:,None],conv_Q,border_mode='half') # batch x dim x SeqL x 1
        cover_F = cover_F.dimshuffle(1,2,0,3) # dim x SeqL x batch x 1
        cover_F = cover_F.reshape([cover_F.shape[0],cover_F.shape[1],cover_F.shape[2]])
        assert cover_F.ndim == 3, \
            'Output of conv must be 3-d: #dim x SeqL x batch'
        cover_F = cover_F.dimshuffle(1, 2, 0)
        # cover_F must be SeqL x batch x dimctx
        cover_vector = tensor.dot(cover_F, conv_Uf) + conv_b
        # cover_vector = cover_vector * context_mask[:,:,None]

        pctx__ = pctx_ + pstate_[None, :, :] + cover_vector
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha - alpha.max()
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        alpha_past = alpha_past_ + alpha.T
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1, U_nl)+b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T, alpha_past  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')],
                   tparams[_p(prefix, 'conv_Q')],
                   tparams[_p(prefix, 'conv_Uf')],
                   tparams[_p(prefix, 'conv_b')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, alpha_past, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb_dec'] = norm_weight(options['dim_target'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder0',
                                              nin=options['dim_feature'],
                                              dim=options['dim_enc'][0])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r0',
                                              nin=options['dim_feature'],
                                              dim=options['dim_enc'][0])


    hiddenSizes=options['dim_enc']
    for i in range(1,len(hiddenSizes)):
        params = get_layer(options['encoder'])[0](options, params,
                                                  prefix='encoder'+str(i),
                                                  nin=hiddenSizes[i-1]*2,
                                                  dim=hiddenSizes[i])
        params = get_layer(options['encoder'])[0](options, params,
                                                  prefix='encoder_r'+str(i),
                                                  nin=hiddenSizes[i-1]*2,
                                                  dim=hiddenSizes[i])
    ctxdim = 2 * hiddenSizes[-1]

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim_dec'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim_dec'],
                                              dimctx=ctxdim,
                                              dimatt=options['dim_attention'])
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_gru',
                                nin=options['dim_dec'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word']/2,
                                nout=options['dim_target'])

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.tensor3('x', dtype='float32')
    x_mask_original = tensor.matrix('x_mask_original', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    a_original = tensor.tensor3('a_original', dtype='float32')
    a_mask_original = tensor.tensor3('a_mask_original', dtype='float32')

    x_mask = x_mask_original
    a = a_original # SeqX * batch * SeqY
    a_mask = a_mask_original # SeqX * batch * SeqY
    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask_original[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    h=x
    hr=xr
    hidden_sizes=options['dim_enc']

    for i in range(len(hidden_sizes)):
        proj = get_layer(options['encoder'])[1](tparams, h, options,
                                                prefix='encoder'+str(i),
                                                mask=x_mask)
        # word embedding for backward rnn (source)
        projr = get_layer(options['encoder'])[1](tparams, hr, options,
                                                 prefix='encoder_r'+str(i),
                                                 mask=xr_mask)

        h=concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
        if options['down_sample'][i]==1:
            h=h[0::2]
            x_mask=x_mask[0::2]
            xr_mask=x_mask[::-1]
            a=a[0::2]
            a_mask=a_mask[0::2]
        hr=h[::-1]

    # a -- SeqX * batch * SeqY
    a = a / (tensor.sum(a, axis=0, keepdims=True) + numpy.array(1e-20).astype('float32'))

    # context will be the concatenation of forward and backward rnns
    ctx = h

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb) # the 0 idx is <eos>!!
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2] # SeqY * batch * SeqX while a -- SeqX * batch * SeqY
    #dec_alphas = proj[2].dimshuffle(2, 1, 0) + tensor.alloc(1e-20, a.shape[0], a.shape[1], a.shape[2]) # SeqX * batch * SeqY
    dec_alphas = proj[2].dimshuffle(2, 1, 0) + numpy.array(1e-20).astype('float32')
    cost_alphas = - a * tensor.log(dec_alphas) * a_mask

    # compute word probabilities
    logit_gru = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_gru', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = logit_gru+logit_prev+logit_ctx

    # maxout 2
    # maxout layer
    shape = logit.shape
    shape2 = tensor.cast(shape[2] / 2, 'int64')
    shape3 = tensor.cast(2, 'int64')
    logit = logit.reshape([shape[0],shape[1], shape2, shape3]) # seq*batch*256 -> seq*batch*128*2
    logit=logit.max(3) # seq*batch*128

    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options, 
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape # (seqL*batch, dim_target)
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]])) # (seqL*batch, dim_target)

    # cost
    cost = tensor.nnet.categorical_crossentropy(probs, y.flatten()) # x is a vector,each value is a 1-of-N position 
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0) + options['gamma'] * cost_alphas.sum(axis=(0,2))

    return trng, use_noise, x, x_mask_original, y, y_mask, a_original, a_mask_original, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.tensor3('x', dtype='float32')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    h=x
    hr=xr
    hidden_sizes=options['dim_enc']

    for i in range(len(hidden_sizes)):
        proj = get_layer(options['encoder'])[1](tparams, h, options,
                                                prefix='encoder'+str(i))
        # word embedding for backward rnn (source)
        projr = get_layer(options['encoder'])[1](tparams, hr, options,
                                                 prefix='encoder_r'+str(i))

        h=concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
        if options['down_sample'][i]==1:
            h=h[0::2]
        hr=h[::-1]

    ctx = h
    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    alpha_past = tensor.matrix('alpha_past', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state, alpha_past = alpha_past)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]
    next_alpha_past = proj[3]

    logit_gru = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_gru', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = logit_gru+logit_prev+logit_ctx

    # maxout layer
    shape = logit.shape
    shape1 = tensor.cast(shape[1] / 2, 'int64')
    shape2 = tensor.cast(2, 'int64')
    logit = logit.reshape([shape[0], shape1, shape2]) # batch*256 -> batch*128*2
    logit=logit.max(2) # batch*500


    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    inps = [y, ctx, init_state, alpha_past]
    outs = [next_probs, next_sample, next_state, next_alpha_past]
    f_next = theano.function(inps, outs, name='f_next', profile=profile,on_unused_input='ignore')
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator
    SeqL = x.shape[0]
    hidden_sizes=options['dim_enc']
    for i in range(len(hidden_sizes)):
        if options['down_sample'][i]==1:
            SeqL = math.ceil(SeqL / 2.)
    next_alpha_past = 0.0 * numpy.ones((1, int(SeqL))).astype('float32') # start position
    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state, next_alpha_past]
        ret = f_next(*inps)
        next_p, next_w, next_state, next_alpha_past = ret[0], ret[1], ret[2], ret[3]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_alpha_past = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                new_hyp_alpha_past.append(copy.copy(next_alpha_past[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_alpha_past = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0: # <eol>
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    hyp_alpha_past.append(new_hyp_alpha_past[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            next_alpha_past = numpy.array(hyp_alpha_past)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=False):
    probs = []

    n_done = 0

    for x, y, a in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask, a, a_mask = prepare_data(options, x, y, a)

        pprobs = f_log_probs(x, x_mask, y, y_mask, a, a_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            #ipdb.set_trace()
            print 'probs nan'

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])

    print 'total words/phones',len(lexicon)
    return lexicon


def apply_adative_noise(tparams,options,trng,grads,num_examples):
    log_sigma_scale = 2048.0
    init_sigma = 1.0e-12
    model_cost_coefficient=options['model_cost_coeff']
    p_noisy=[]
    Beta=[]
    tparams_p_u=OrderedDict()
    tparams_p_ls2=OrderedDict()
    #tparams_p_s2=OrderedDict()
    for k,p in tparams.iteritems():
        p_u = theano.shared(p.get_value(),name='%s_miu' % k)  # miu
        p_ls2 = theano.shared((numpy.zeros_like(p.get_value()) +
                               numpy.log(init_sigma) * 2. / log_sigma_scale
                               ).astype(dtype=numpy.float32) ,name='%s_sigma' % k) # log_(sigma^2)
        p_s2 = tensor.exp(p_ls2 * log_sigma_scale) # sigma^2

        Beta.append((p_u,p_ls2,p_s2))
        tparams_p_u[k]=p_u
        tparams_p_ls2[k]=p_ls2
        #tparams_p_s2[k]=p_s2
        #p_noisy_tmp = p_u + trng.normal(size=p.get_value().shape) * tensor.sqrt(p_s2)
        #p_noisy.append(p_noisy_tmp)
    


    #  compute the prior mean and variation
    temp_sum = 0.0
    temp_param_count = 0.0
    for p_u, unused_p_ls2, unused_p_s2 in Beta:
        temp_sum = temp_sum + p_u.sum()
        temp_param_count = temp_param_count + p_u.shape.prod()

    prior_u = tensor.cast(temp_sum / temp_param_count, 'float32')

    temp_sum = 0.0
    for p_u, unused_ls2, p_s2 in Beta:
        temp_sum = temp_sum + (p_s2).sum() + (((p_u-prior_u)**2).sum())

    prior_s2 = tensor.cast(temp_sum/temp_param_count, 'float32')
    

    # update miu and sigma gradient with w's grads ## grads is a list
    new_grads_miu = []
    new_grads_sigma = []

    # Warning!!! ????? maybe error????
    # This only works for batch size 1 (we want that the sum of squares
    # be the square of the sum!
    #
    #diag_hessian_estimate = [g**2 for g in grads]

    for (p_u, p_ls2, p_s2),p_grad in zip(Beta,grads):
        p_u_grad = (model_cost_coefficient * (p_u - prior_u) /
                    (num_examples*prior_s2) + p_grad)

        p_ls2_grad = (numpy.float32(model_cost_coefficient *
                                    0.5 / num_examples * log_sigma_scale) *
                      (p_s2/prior_s2 - 1.0) +
                      (0.5*log_sigma_scale) * p_s2 * (p_grad**2)
                      )
        new_grads_miu.append(p_u_grad)
        new_grads_sigma.append(p_ls2_grad)



    # return

    # add noise to weight
    p_add_noise=[]
    for p,p_ls2 in zip(itemlist(tparams),itemlist(tparams_p_ls2)):
        p_add_noise.append((p,p+trng.normal(size=p.get_value().shape) * tensor.sqrt(tensor.exp(p_ls2 * log_sigma_scale))))
    f_apply_noise_to_weight = theano.function([], [], updates=p_add_noise, profile=profile)



    # restore weight
    copy_weight=[]
    for p,p_u in zip(itemlist(tparams),itemlist(tparams_p_u)):
        copy_weight.append((p,p_u))
    f_copy_weight = theano.function([],[],updates=copy_weight, profile=profile)



    return f_apply_noise_to_weight,f_copy_weight,new_grads_miu,new_grads_sigma,tparams_p_u,tparams_p_ls2 #,tparams_p_s2



def train(dim_word=100,  # word vector dimensionality
          dim_enc=1000,
          dim_dec=1000,
          down_sample=0,
          dim_attention=500,
          dim_coverage=5,
          kernel_coverage=121,
          encoder='gru',
          decoder='gru_cond',
          patience=4,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=1e-8,  # learning rate
          model_cost_coeff=0.1, # model cost coefficient for adaptive noise
          gamma=1e-3,  # alphas gamma
          dim_target=62,  # source vocabulary size
          dim_feature=123,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=['feature.pkl',
                    'label.txt',
                    'align.pkl'],
          valid_datasets=['feature_valid.pkl', 
                          'label_valid.txt'],
          dictionaries=['lexicon.txt'],
          valid_output=['decode.txt'],
          valid_result=['result.txt'],
          use_dropout=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them

    worddicts = load_dict(dictionaries[0])
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)

    print 'Loading data'
    train = dataIterator(datasets[0], datasets[1], datasets[2],
                         worddicts,
                         batch_size=batch_size,maxlen=maxlen)
    valid,valid_uid_list = dataIterator_valid(valid_datasets[0], valid_datasets[1],
                         worddicts,
                         batch_size=valid_batch_size,maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, a, a_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, a, a_mask]

    print 'Buliding sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # apply_adaptive_noise and update gradient for adadelta
    f_apply_noise_to_weight,f_copy_weight,new_grads_miu,new_grads_sigma, \
                tparams_p_u,tparams_p_ls2=apply_adative_noise(tparams,model_options,trng,grads,2*8835)

    # compile the optimizer, the actual computational graph is compiled here
    print 'Building optimizer for miu and sigma'
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update_miu, f_update_sigma = eval(optimizer)(lr, tparams_p_u, tparams_p_ls2, new_grads_miu, new_grads_sigma, inps, cost)
    print 'Done'

    
    
    # print model parameters
    print "Model params:\n{0}".format(
            pprint.pformat(sorted([p for p in params])))
    # end



    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train)
    if saveFreq == -1:
        saveFreq = len(train)
    if sampleFreq == -1:
        sampleFreq = len(train)

    uidx = 0
    estop = False
    halfLrFlag = 0
    bad_counter = 0
    ud_s = 0
    ud_epoch = 0
    cost_s = 0.
    for eidx in xrange(max_epochs):
        n_samples = 0

        random.shuffle(train) # shuffle data
        ud_epoch_start = time.time()

        for x, y, a in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            ud_start = time.time()

            x, x_mask, y, y_mask, a, a_mask = prepare_data(model_options, x, y, a, maxlen=maxlen)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            # compute cost, grads and copy grads to shared variables
            f_apply_noise_to_weight() # add noise to tparams
            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask, a, a_mask)
            cost_s += cost
            f_update_miu(lrate) # update p_u
            f_update_sigma(lrate) # update p_sigma
            f_copy_weight() # copy weight from tparams_p_u to tparams

            ud = time.time() - ud_start
            ud_s += ud
            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                ud_s /= 60.
                cost_s /= dispFreq
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost_s, 'UD ', ud_s, 'epson ',lrate, 'bad_counter', bad_counter
                ud_s = 0
                cost_s = 0.

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                fpp_sample=open(valid_output[0],'w')
                valid_count_idx=0
                # FIXME: random selection?
                for x,y in valid:
                    for xx in x:
                        xx_pad = numpy.zeros((xx.shape[0]+1,xx.shape[1]), dtype='float32')
                        xx_pad[:xx.shape[0],:] = xx
                        stochastic = False
                        sample, score = gen_sample(tparams, f_init, f_next,
                                                   xx_pad[:, None, :],
                                                   model_options, trng=trng, k=10,
                                                   maxlen=1000,
                                                   stochastic=stochastic,
                                                   argmax=False)
                        
                        if stochastic:
                            ss = sample
                        else:
                            score = score / numpy.array([len(s) for s in sample])
                            ss = sample[score.argmin()]

                        fpp_sample.write(valid_uid_list[valid_count_idx])
                        valid_count_idx=valid_count_idx+1
                        for vv in ss:
                            if vv == 0: # <eol>
                                break
                            fpp_sample.write(' '+worddicts_r[vv])
                        fpp_sample.write('\n')
                fpp_sample.close()
                print 'valid set decode done'
                ud_epoch = (time.time() - ud_epoch_start) / 60.
                print 'cost time ... ', ud_epoch



            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                #valid_errs = pred_probs(f_log_probs, prepare_data,
                                        #model_options, valid)
                #valid_err_cost = valid_errs.mean()
                

                # compute wer
                os.system('python compute-wer.py ' + valid_output[0] + ' ' + valid_datasets[1] + ' ' + valid_result[0])
                fpp=open(valid_result[0])
                stuff=fpp.readlines()
                fpp.close()
                m=re.search('WER (.*)\n',stuff[0])
                valid_per=100. * float(m.group(1))
                m=re.search('ExpRate (.*)\n',stuff[1])
                valid_sacc=100. * float(m.group(1))
                #valid_err=0.7*valid_per-0.3*valid_sacc
                valid_err=valid_per
                history_errs.append(valid_err)

                if uidx/validFreq == 0 or valid_err <= numpy.array(history_errs).min(): # the first time valid or worse model
                    best_p = unzip(tparams)
                    bad_counter = 0
                # if len(history_errs) > patience and valid_err >= \
                #         numpy.array(history_errs)[:-patience].min():
                #     bad_counter += 1
                #     if bad_counter > patience:
                #         print 'Early Stop!'
                #         estop = True
                #         break
                if uidx/validFreq != 0 and valid_err > numpy.array(history_errs).min():
                    bad_counter += 1
                    if bad_counter > patience:
                        if halfLrFlag==2:
                            print 'Early Stop!'
                            estop = True
                            break
                        else:
                            print 'Lr decay and retrain!'
                            bad_counter = 0
                            lrate /= 10
                            params = best_p
                            halfLrFlag += 1

                if numpy.isnan(valid_err):
                    #ipdb.set_trace()
                    print 'valid_err nan'

                print 'Valid WER: %.2f%%, ExpRate: %.2f%%' % (valid_per,valid_sacc)

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    #valid_err = pred_probs(f_log_probs, prepare_data,
                           #model_options, valid).mean()

    #print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                **params)

    #return valid_err
    return


if __name__ == '__main__':
    pass
