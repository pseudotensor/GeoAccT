import theano
from theano import tensor
from theano import scan
import numpy as np


def np_linear_cg(compute_Gv, bs, rtol=1e-6, maxit=1000, damp=0):
    rs = [x for x in bs]
    ps = [x for x in rs]
    xs = [np.zeros_like(x) for x in rs]
    rsold = sum(np.sum(x * x) for x in rs)

    for i in range(maxit):
        _Aps = compute_Gv(ps)
        Aps = [x + damp * y for x, y in zip(_Aps, ps)]
        alpha = rsold / sum(np.sum(x * y) for x, y in zip(Aps, ps))
        xs = [x + alpha * p for x, p in zip(xs, ps)]
        rs = [r - alpha * Ap for r, Ap in zip(rs, Aps)]
        rsnew = sum(np.sum(r * r) for r in rs)
        ps = [r + rsnew / rsold * p for r, p in zip(rs, ps)]
        rsold = rsnew
    return xs


####################################################################################################

def linear_cg(compute_Gv, bs, *x0s, rtol=5e-4, maxit=100, damp=0):
    """
    Adapted by Jonathan C. McKinney from https://github.com/pascanur/theano_optimize
    """
    n_params = len(bs)

    def loop(rsold, *args):
        ps = args[:n_params]
        rs = args[n_params:2 * n_params]
        xs = args[2 * n_params:]
        _Aps = compute_Gv(ps)
        Aps = [x + damp * y for x, y in zip(_Aps, ps)]
        alpha = rsold / sum((x * y).sum() for x, y in zip(Aps, ps))
        xs = [x + alpha * p for x, p in zip(xs, ps)]
        rs = [r - alpha * Ap for r, Ap in zip(rs, Aps)]
        rsnew = sum((r * r).sum() for r in rs)
        ps = [r + rsnew / rsold * p for r, p in zip(rs, ps)]
        return [rsnew] + ps + rs + xs, \
               theano.scan_module.until(abs(rsnew) < rtol)

    _x0s = list(x0s[:])
    if _x0s is None:
        _r0s = bs[:]
    else:
        Ax = compute_Gv(_x0s)
        _r0s = [x - y for x, y in zip(bs, Ax)]
    _p0s = _r0s[:]
    _x0s = [tensor.zeros_like(x) for x in bs] if _x0s is None else _x0s
    _rsold = sum(tensor.sum(r * r) for r in _r0s)
    outs, updates = scan(loop,
                         outputs_info=[_rsold] + _p0s + _r0s + _x0s,
                         name='linear_conjugate_gradient',
                         n_steps=maxit)
    ans = outs[1 + 2 * n_params:]
    return [x[-1] for x in ans]


def linear_cg_precond(compute_Gv, bs, Msz, *x0s, rtol=5e-4, maxit=100):
    """
    Adapted by Jonathan C. McKinney from https://github.com/pascanur/theano_optimize
    """
    n_params = len(bs)

    def loop(rsold, *args):
        ps = args[:n_params]
        rs = args[n_params:2 * n_params]
        xs = args[2 * n_params:]
        Aps = compute_Gv(ps)
        alpha = rsold / sum((x * y).sum() for x, y in zip(Aps, ps))
        xs = [x + alpha * p for x, p in zip(xs, ps)]
        rs = [r - alpha * Ap for r, Ap, in zip(rs, Aps)]
        zs = [r / z for r, z in zip(rs, Msz)]
        rsnew = sum((r * z).sum() for r, z in zip(rs, zs))
        ps = [z + rsnew / rsold * p for z, p in zip(zs, ps)]
        return [rsnew] + ps + rs + xs, \
               theano.scan_module.until(abs(rsnew) < rtol)

    _x0s = list(x0s[:])
    Ax = compute_Gv(_x0s)
    _r0s = [x - y for x, y in zip(bs, Ax)]
    _p0s = [x / z for x, z in zip(_r0s, Msz)]

    _rsold = sum((r * r / z).sum() for r, z in zip(_r0s, Msz))

    outs, updates = scan(loop,
                         outputs_info=[_rsold] + _p0s + _r0s + _x0s,
                         n_steps=maxit,
                         name='linear_conjugate_gradient_precond')
    fxs = outs[1 + 2 * n_params:]
    return [x[-1] for x in fxs]


####################################################################################################

def linear_cg_original(compute_Gv, bs, rtol=1e-6, maxit=1000, damp=0, floatX=None, profile=0):
    """
    assume all are lists all the time
    Reference:
        http://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    n_params = len(bs)

    def loop(rsold, *args):
        ps = args[:n_params]
        rs = args[n_params:2 * n_params]
        xs = args[2 * n_params:]
        _Aps = compute_Gv(*ps)
        Aps = [x + damp * y for x, y in zip(_Aps, ps)]
        alpha = rsold / sum((x * y).sum() for x, y in zip(Aps, ps))
        xs = [x + alpha * p for x, p in zip(xs, ps)]
        rs = [r - alpha * Ap for r, Ap, in zip(rs, Aps)]
        rsnew = sum((r * r).sum() for r in rs)
        ps = [r + rsnew / rsold * p for r, p in zip(rs, ps)]
        return [rsnew] + ps + rs + xs, \
               theano.scan_module.until(abs(rsnew) < rtol)

    r0s = bs
    _p0s = [tensor.unbroadcast(tensor.shape_padleft(x), 0) for x in r0s]
    _r0s = [tensor.unbroadcast(tensor.shape_padleft(x), 0) for x in r0s]
    _x0s = [tensor.unbroadcast(tensor.shape_padleft(
        tensor.zeros_like(x)), 0) for x in bs]
    rsold = sum((r * r).sum() for r in r0s)
    _rsold = tensor.unbroadcast(tensor.shape_padleft(rsold), 0)
    outs, updates = scan(loop,
                         states=[_rsold] + _p0s + _r0s + _x0s,
                         n_steps=maxit,
                         mode=theano.Mode(linker='cvm'),
                         name='linear_conjugate_gradient',
                         profile=profile)
    fxs = outs[1 + 2 * n_params:]
    return [x[0] for x in fxs]


def linear_cg_precond_original(compute_Gv, bs, Msz, rtol=1e-16, maxit=100000, floatX=None):
    """
    assume all are lists all the time
    Reference:
        http://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    n_params = len(bs)

    def loop(rsold, *args):
        ps = args[:n_params]
        rs = args[n_params:2 * n_params]
        xs = args[2 * n_params:]
        Aps = compute_Gv(*ps)
        alpha = rsold / sum((x * y).sum() for x, y in zip(Aps, ps))
        xs = [x + alpha * p for x, p in zip(xs, ps)]
        rs = [r - alpha * Ap for r, Ap, in zip(rs, Aps)]
        zs = [r / z for r, z in zip(rs, Msz)]
        rsnew = sum((r * z).sum() for r, z in zip(rs, zs))
        ps = [z + rsnew / rsold * p for z, p in zip(zs, ps)]
        return [rsnew] + ps + rs + xs,

    theano.scan_module.until(abs(rsnew) < rtol)

    r0s = bs
    _p0s = [tensor.unbroadcast(tensor.shape_padleft(x / z), 0) for x, z in zip(r0s, Msz)]
    _r0s = [tensor.unbroadcast(tensor.shape_padleft(x), 0) for x in r0s]
    _x0s = [tensor.unbroadcast(tensor.shape_padleft(
        tensor.zeros_like(x)), 0) for x in bs]
    rsold = sum((r * r / z).sum() for r, z in zip(r0s, Msz))
    _rsold = tensor.unbroadcast(tensor.shape_padleft(rsold), 0)
    outs, updates = scan(loop,
                         states=[_rsold] + _p0s + _r0s + _x0s,
                         n_steps=maxit,
                         mode=theano.Mode(linker='c|py'),
                         name='linear_conjugate_gradient',
                         profile=0)
    fxs = outs[1 + 2 * n_params:]
    return [x[0] for x in fxs]
