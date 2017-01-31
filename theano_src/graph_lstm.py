_doc__ = """Implementation of Tree LSTMs described in http://arxiv.org/abs/1503.00075"""

import tree_rnn

import theano
import numpy as np
from theano import tensor as T
#from memory_profiler import profile

class ArrayInit(object):
    normal = 'normal'
    onesided_uniform = 'onesided_uniform'
    twosided_uniform = 'twosided_uniform'
    ortho = 'ortho'
    zero = 'zero'
    unit = 'unit'
    ones = 'ones'
    fromfile = 'fromfile'
    def __init__(self, option,
            multiplier=0.01,
            matrix=None,
            word2idx=None):
        self.option = option
        self.multiplier = multiplier
        self.matrix_filename = None
        self.matrix = self._matrix_reader(matrix, word2idx)
        if self.matrix is not None:
            self.multiplier = 1
        return

    def _matrix_reader(self, matrix, word2idx):
        if type(matrix) is str:
            self.matrix_filename = matrix
            assert os.path.exists(matrix), "File %s not found"%matrix
            matrix = read_matrix_from_file(matrix, word2idx)
            return matrix
        else:
            return None

    def initialize(self, *xy, **kwargs):
        if self.option == ArrayInit.normal:
            M = np.random.randn(*xy)
        elif self.option == ArrayInit.onesided_uniform:
            M = np.random.rand(*xy)
        elif self.option == ArrayInit.twosided_uniform:
            M = np.random.uniform(-1.0, 1.0, xy)
        elif self.option == ArrayInit.ortho:
            f = lambda dim: np.linalg.svd(np.random.randn(dim, dim))[0]
            if int(xy[1]/xy[0]) < 1 and xy[1]%xy[0] != 0:
                raise ValueError(str(xy))
            M = np.concatenate(tuple(f(xy[0]) for _ in range(int(xy[1]/xy[0]))),
                    axis=1)
            assert M.shape == xy
        elif self.option == ArrayInit.zero:
            M = np.zeros(xy)
        elif self.option in [ArrayInit.unit, ArrayInit.ones]:
            M = np.ones(xy)
        elif self.option == ArrayInit.fromfile:
            assert isinstance(self.matrix, np.ndarray)
            M = self.matrix
        else:
            raise NotImplementedError
        #self.multiplier = (kwargs['multiplier']
        multiplier = (kwargs['multiplier']
            if ('multiplier' in kwargs
                    and kwargs['multiplier'] is not None)
                else self.multiplier)
        #return (M*self.multiplier).astype(config.floatX)
        return (M*multiplier).astype(config.floatX)

    def __repr__(self):
        mults = ', multiplier=%s'%((('%.3f'%self.multiplier)
            if type(self.multiplier) is float
            else str(self.multiplier)))
        mats = ((', matrix="%s"'%self.matrix_filename)
                if self.matrix_filename is not None
                else '')
        return "ArrayInit(ArrayInit.%s%s%s)"%(self.option, mults, mats)


class ChildAvgGraphLSTM(tree_rnn.TreeRNN):

    def _check_input(self, tree):
        assert np.all((tree[:, 0] <= np.arange(len(tree))) |
                      (tree[:, 0] == -1))
        assert np.all((tree[:, 1] <= np.arange(len(tree))) |
                      (tree[:, 1] == -1))
   
    #@profile
    def create_recursive_unit_weighted(self):
        self.W = theano.shared(
                np.concatenate([ArrayInit(ArrayInit.twosided_uniform).initialize([self.hidden_dim, self.emb_dim], multiplier=1) for i in range(4)]), 
                name='GraphLSTM_W')
        self.U = theano.shared(
                np.concatenate([ArrayInit(ArrayInit.ortho).initialize([self.hidden_dim, self.hidden_dim, self.arc_type], multiplier=1) for i in range(4)]), 
                name='GraphLSTM_U')
        self.b = theano.shared(
                np.concatenate([ArrayInit(ArrayInit.zero).initialize([self.hidden_dim]) for i in range(4)]), 
                name='GraphLSTM_p')
        self.params.extend([self.W, self.U, self.b]) 
        
        def __slice(matrix, row_idx, stride):
            return matrix[row_idx*stride: (row_idx+1)*stride]

        ''' Shapes: (sent_len,), (hidden_dim, sent_len, arc_type), (hidden_dim, sent_len), (sent_len, arc_type)'''
        #@profile
        def unit(parent_x, child_h, child_c, child_exists):
            #h_tilde = T.dot(child_h, child_exists) / child_exists.sum() 
            h_tilde = T.sum(child_h, axis=1) / child_exists.sum() #T.cast(, theano.config.floatX) 
            i = T.nnet.sigmoid(
                    T.dot(__slice(self.W, 0, self.hidden_dim), parent_x) 
                    + T.tensordot(__slice(self.U, 0, self.hidden_dim), h_tilde, axes=((1,2), (0,1)))
                    + __slice(self.b, 0, self.hidden_dim))
            o = T.nnet.sigmoid(
                    T.dot(__slice(self.W, 1, self.hidden_dim), parent_x) 
                    #+ (__slice(self.U, 1, self.hidden_dim) * h_tilde).sum(axis=(1,2))
                    + T.tensordot(__slice(self.U, 1, self.hidden_dim), h_tilde, axes=((1,2), (0,1)))
                    + __slice(self.b, 1, self.hidden_dim))
            u = T.tanh(
                    T.dot(__slice(self.W, 2, self.hidden_dim), parent_x) 
                    #+ (__slice(self.U, 2, self.hidden_dim) * h_tilde).sum(axis=(1,2))
                    + T.tensordot(__slice(self.U, 2, self.hidden_dim), h_tilde, axes=((1,2), (0,1)))
                    + __slice(self.b, 2, self.hidden_dim))
            # Need to be careful here, ideally, f should have dimension (self.hidden_dim, #precedants), my current implementations seems only be able to make it (self.hidden_dim, sent_len)
            f = T.nnet.sigmoid(
                    T.dot(__slice(self.W, 3, self.hidden_dim), parent_x)[:, None] 
                    #+ (__slice(self.U, 3, self.hidden_dim)[:, :, None, :] * child_h).sum(axis=[1,3]) 
                    #+ (__slice(self.U, 3, self.hidden_dim)[:, :, None, :] * child_h).sum(axis=1) 
                    + T.dot(__slice(self.U, 3, self.hidden_dim).dimshuffle(0,2,1), child_h.sum(axis=2)).sum(axis=1) 
                    #+ T.tensordot(__slice(self.U, 3, self.hidden_dim), child_h, axes=((1,2), (0,2))) 
                    + __slice(self.b, 3, self.hidden_dim)[:, None]) / child_exists.sum() 

            c = i * u + T.sum(f * child_c, axis=1) 
            h = o * T.tanh(c)
            return h, c

        return unit


    #@profile
    def create_recursive_unit(self):
        '''self.W_i = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_i = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_i = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_f = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_f = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_f = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_o = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_o = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_o = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_u = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_u = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_u = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_u, self.U_u, self.b_u])
        '''
        self.W = theano.shared(
                np.concatenate([self.init_matrix([self.hidden_dim, self.emb_dim]) for i in range(4)]), 
                name='GraphLSTM_W')
        self.U = theano.shared(
                np.concatenate([self.init_matrix([self.hidden_dim, self.hidden_dim]) for i in range(4)]), 
                name='GraphLSTM_U')
        self.b = theano.shared(
                np.concatenate([self.init_vector([self.hidden_dim]) for i in range(4)]), 
                name='GraphLSTM_p')
        self.params.extend([self.W, self.U, self.b]) 
        
        def __slice(matrix, row_idx, stride):
            return matrix[row_idx*stride: (row_idx+1)*stride]

        ''' Shapes: (sent_len,), (hidden_dim, sent_len), (hidden_dim, sent_len), (1, sent_len)'''
        #@profile
        def unit(parent_x, child_h, child_c, child_exists):
            h_tilde = T.sum(child_h, axis=1) / child_exists.sum() #T.cast(, theano.config.floatX) 
            i = T.nnet.sigmoid(T.dot(__slice(self.W, 0, self.hidden_dim), parent_x) + T.dot(__slice(self.U, 0, self.hidden_dim), h_tilde) + __slice(self.b, 0, self.hidden_dim))
            o = T.nnet.sigmoid(T.dot(__slice(self.W, 1, self.hidden_dim), parent_x) + T.dot(__slice(self.U, 1, self.hidden_dim), h_tilde) + __slice(self.b, 1, self.hidden_dim))
            u = T.tanh(T.dot(__slice(self.W, 2, self.hidden_dim), parent_x) + T.dot(__slice(self.U, 2, self.hidden_dim), h_tilde) + __slice(self.b, 2, self.hidden_dim))

            f = (T.nnet.sigmoid(
                    T.dot(__slice(self.W, 3, self.hidden_dim), parent_x).dimshuffle(0, 'x') +
                    #T.dot(child_h, self.U_f.T) +
                    T.dot(__slice(self.U, 3, self.hidden_dim), child_h) +
                    __slice(self.b, 3, self.hidden_dim).dimshuffle(0, 'x'))) / child_exists.sum()
            c = i * u + T.sum(f * child_c, axis=1)
            h = o * T.tanh(c)
            return h, c

        return unit

    
    def create_init_unit(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))
        def unit(leaf_x):
            return self.recursive_unit(
                leaf_x,
                dummy,
                dummy,
                dummy.sum(axis=1))
        return unit

    
    #@profile
    def compute_tree(self, emb_x, child_exists):
        self.recursive_unit = self.create_recursive_unit() #_weighted()
        num_nodes = emb_x.shape[0]  # num internal nodes
        #self.init_unit = self.create_init_unit()

        #def dummy_scan(x):
        #    return theano.shared(np.zeros(self.hidden_dim).astype(theano.config.floatX)), theano.shared(np.zeros(self.hidden_dim).astype(theano.config.floatX))
        # compute leaf hidden states
        #(leaf_h, leaf_c), _ = theano.map(
        #    fn=dummy_scan,  #dummy_scan,
        #    sequences=[emb_x])  #[:num_leaves]])


        leaf_h = T.zeros((self.hidden_dim, num_nodes)).astype(theano.config.floatX)
        leaf_c = T.zeros((self.hidden_dim, num_nodes)).astype(theano.config.floatX)
       
        print 'shapes:', self.emb_dim, self.hidden_dim, self.arc_type
        # use recurrence to compute internal node hidden states
        #@profile
        def _recurrence(cur_emb, child_exists, t, node_h, node_c, last_h):
        #def _recurrence(cur_emb, node_info, child_exists, t, node_h, node_c, last_h):
            #child_exists = node_info > -1
            #child_h = node_h[node_info - child_exists * t] * child_exists.dimshuffle(0, 'x')
            #child_c = node_c[node_info - child_exists * t] * child_exists.dimshuffle(0, 'x')
            ''' For typed arcs:
                (hidden_dim, sent_len) * (sent_len, type)
                We should be careful here, we want to know the precedant and their type.
                We got tensors with dimension (hidden_dim, sent_len, arc_type.)'''
            #child_h = node_h[:, :, None] * child_exists  #.sum(axis=1)
            #child_c = node_c * child_exists.sum(axis=1)
            ''' For non-typed arcs:
                shapes: (hidden_dim, sent_len) * (sent_len,)'''
            child_h = node_h * child_exists
            child_c = node_c * child_exists
            ''' Shapes: (sent_len,), (hidden_dim, sent_len), (hidden_dim, sent_len), (1, sent_len)'''
            curr_h, curr_c = self.recursive_unit(cur_emb, child_h, child_c, child_exists)  
            node_h = T.set_subtensor(node_h[:,t], curr_h)#.reshape(1, self.hidden_dim)) 
            node_c = T.set_subtensor(node_c[:,t], curr_c)#.reshape(1, self.hidden_dim)) 
            return node_h, node_c, curr_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[leaf_h, leaf_c, dummy],
            sequences=[emb_x, child_exists, T.arange(num_nodes)],
            n_steps=num_nodes)

        return parent_h
