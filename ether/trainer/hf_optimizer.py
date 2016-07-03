# Author: Nicolas Boulanger-Lewandowski
# University of Montreal, 2012-2013
# Modified by Ether Wei 2016

'''
Hessian free optimizer has relatively limited use
Ususally it's applied to the RNN
Due to theano implementation details, some of the ops doesn't support the T.Rop, like downsample used in traditional Lenet
Also, some un-supervised models are not fitted into it either
'''

from core import *
import numpy, sys
import theano
import theano.tensor as T

def gauss_newton_product(cost, params, v, s):  # this computes the product Gv = J'HJv (G is the Gauss-Newton matrix)
    if s.ndim != 0:
        s = T.sum(s)
    Jv = T.Rop(s, params, v)
    HJv = T.grad(T.sum(T.grad(cost, s)*Jv), s, consider_constant=[Jv], disconnected_inputs='ignore')
    Gv = T.grad(T.sum(HJv*s), params, consider_constant=[HJv, Jv], disconnected_inputs='ignore')
    Gv = map(T.as_tensor_variable, Gv)  # for CudaNdarray
    return Gv


class HessianFreeOptimizer(optimizerBase):
    def __init__(self, h=None, ha=None,
                 initial_lambda=0.1, mu=0.03, preconditioner=False,
                 max_cg_iterations=250, num_updates=100, patience=numpy.inf):
        '''
        Constructs and compiles the necessary Theano functions.

        :type initial_lambda: float
        :param initial_lambda: Initial value of the Tikhonov damping coefficient.
        :type mu: float
        :param mu: Coefficient for structural damping.
        :type preconditioner: Boolean
        :param preconditioner: Whether to use Martens' preconditioner.
        :type max_cg_iterations: int
        :param max_cg_iterations: CG stops after this many iterations regardless of the stopping criterion.
        :type num_updates: int
        :param num_updates: Training stops after this many parameter updates regardless of `patience`.

        :type h: Theano variable or None
        :param h: Structural damping is applied to this variable (typically the hidden units of an RNN).
        :type ha: Theano variable or None
        :param ha: Symbolic variable that implicitly defines the Gauss-Newton matrix for the structural damping term
                    (typically the activation of the hidden layer). If None, it will be set to `h`.
        '''

        self.h = h
        self.ha = ha
        self.lambda_ = initial_lambda
        self.mu = mu
        self.preconditioner = preconditioner
        self.max_cg_iterations = max_cg_iterations
        self.num_updates = num_updates
        self.patience = patience

    def init_train(self):
        if self.is_supervise():
            inputs = [self.get_inputTensor(), self.get_targetTensor()]
        else: inputs = [self.get_inputTensor()]
        self.params = self.get_params() #calling get_params() everytime will result in memory leak
        self.shapes = [i.get_value().shape for i in self.params]
        self.sizes = map(numpy.prod, self.shapes)
        self.positions = numpy.cumsum([0] + self.sizes)[:-1]

        g = [gp[0] for gp in self.get_gparams()]
        g = map(T.as_tensor_variable, g)  # for CudaNdarray
        self.f_gc = theano.function(inputs, g + [self.get_cost()], on_unused_input='ignore')  # during gradient computation
        self.f_cost = theano.function(inputs, self.get_cost(), on_unused_input='ignore')  # for quick cost evaluation

        symbolic_types = T.scalar, T.vector, T.matrix, T.tensor3, T.tensor4

        v = [symbolic_types[len(i)]() for i in self.shapes]
        Gv = gauss_newton_product(self.get_cost(), self.params, v, self.get_outputTensor())

        coefficient = T.scalar()  # this is lambda*mu
        if self.h is not None:  # structural damping with cross-entropy
            h_constant = symbolic_types[self.h.ndim]()  # T.Rop does not support `consider_constant` yet, so use `givens`
            structural_damping = coefficient * (-h_constant*T.log(self.h + 1e-10) - (1-h_constant)*T.log((1-self.h) + 1e-10)).sum() / self.h.shape[0]
            if self.ha is None: ha = self.h
            Gv_damping = gauss_newton_product(structural_damping, self.params, v, self.ha)
            Gv = [a + b for a, b in zip(Gv, Gv_damping)]
            givens = {h_constant: self.h}
        else:
            givens = {}

        self.function_Gv = theano.function(inputs + v + [coefficient], Gv, givens=givens,
                                           on_unused_input='ignore')

    def cg(self, b):
        if self.preconditioner:
            M = self.lambda_ * numpy.ones_like(b)
            M += self.list_to_flat(self.f_gc(self.attr, self.tar)[:len(self.params)])**2  #/ self.cg_dataset.number_batches**2
            #print 'precond~%.3f,' % (M - self.lambda_).mean(),
            M **= -0.75  # actually 1/M
            sys.stdout.flush()
        else:
            M = 1.0

        x = self.cg_last_x if hasattr(self, 'cg_last_x') else numpy.zeros_like(b)  # sharing information between CG runs
        r = b - self.stochastic_Gv(x)
        d = M*r
        delta_new = numpy.dot(r, d)
        backtracking = []
        phi = []

        for i in xrange(1, 1 + self.max_cg_iterations):
            q = self.stochastic_Gv(d)
            dq = numpy.dot(d, q)
            #assert dq > 0, 'negative curvature'
            alpha = delta_new / dq
            x = x + alpha*d
            r = r - alpha*q
            s = M*r
            delta_old = delta_new
            delta_new = numpy.dot(r, s)
            d = s + (delta_new / delta_old) * d

            if i >= int(numpy.ceil(1.3**len(backtracking))):
                backtracking.append((x.copy(), i))

            phi_i = -0.5 * numpy.dot(x, r + b)
            phi.append(phi_i)

            k = max(10, i/10)
            if i > k and phi_i < 0 and (phi_i - phi[-k-1]) / phi_i < k*0.0005:
                break

        self.cg_last_x = x.copy()
        j = len(backtracking) - 1
        return backtracking[j] + (i,)

    def flat_to_list(self, vector):
        return [vector[position:position + size].reshape(shape) for shape, size, position in zip(self.shapes, self.sizes, self.positions)]

    def list_to_flat(self, l):
        return numpy.concatenate([i.flatten() for i in l])

    def stochastic_Gv(self, vector, lambda_=None):
        v = self.flat_to_list(vector)
        if lambda_ is None: lambda_ = self.lambda_
        result = lambda_*vector  # Tikhonov damping
        result += self.list_to_flat(self.function_Gv(self.attr, self.tar, v, lambda_*self.mu))
        return result

    def quick_cost(self, delta=0):
        # quickly evaluate objective (costs[0]) over the CG batch
        # for `current params` + delta
        # delta can be a flat vector or a list (else it is not used)
        if isinstance(delta, numpy.ndarray):
            delta = self.flat_to_list(delta)

        if type(delta) in (list, tuple):
            for i, d in zip(self.params, delta):
                i.set_value(i.get_value() + d)

        cost = self.f_cost(self.attr, self.tar)

        if type(delta) in (list, tuple):
            for i, d in zip(self.params, delta):
                i.set_value(i.get_value() - d)

        return cost

    def train_once(self, attr, tar):
        self.attr = attr
        self.tar = tar
        best = [0, numpy.inf, None]  # iteration, cost, params
        first_iteration = 1

        for u in xrange(first_iteration, 1 + self.num_updates):
            print 'update %i/%i,' % (u, self.num_updates),
            sys.stdout.flush()

            gradient = numpy.zeros(sum(self.sizes), dtype=theano.config.floatX)
            cost = []
            result = self.f_gc(self.attr, self.tar)
            gradient += self.list_to_flat(result[:len(self.params)])
            cost.append(result[len(self.params):])

            print 'cost=', numpy.mean(cost, axis=0),
            print 'lambda=%.5f,' % self.lambda_,
            sys.stdout.flush()

            after_cost, flat_delta, backtracking, num_cg_iterations = self.cg(-gradient)
            delta_cost = numpy.dot(flat_delta, gradient + 0.5*self.stochastic_Gv(flat_delta, lambda_=0))  # disable damping
            before_cost = self.quick_cost()
            for i, delta in zip(self.params, self.flat_to_list(flat_delta)):
                i.set_value(i.get_value() + delta)

            rho = (after_cost - before_cost) / delta_cost  # Levenberg-Marquardt
            if rho < 0.25:
                self.lambda_ *= 1.5
            elif rho > 0.75:
                self.lambda_ /= 1.5

            if u - best[0] > self.patience:
                print 'PATIENCE ELAPSED, BAILING OUT'
                break

        if best[2] is None:
            best[2] = [i.get_value().copy() for i in self.params]
        return best[2]
