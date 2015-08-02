import numpy as np
import theano
import theano.tensor as T

import cPickle as pickle
from collections import OrderedDict

class VRAE:
    """This class implements the Variational Recurrent Auto Encoder"""
    def __init__(self, hidden_units_encoder, hidden_units_decoder, features, latent_variables, b1, b2, learning_rate, sigma_init, batch_size):
        self.batch_size = batch_size
        self.hidden_units_encoder = hidden_units_encoder
        self.hidden_units_decoder = hidden_units_decoder
        self.features = features
        self.latent_variables = latent_variables

        self.b1 = theano.shared(np.array(b1).astype(theano.config.floatX), name = "b1")
        self.b2 = theano.shared(np.array(b2).astype(theano.config.floatX), name = "b2")
        self.learning_rate = theano.shared(np.array(learning_rate).astype(theano.config.floatX), name="learning_rate")

        #Initialize all variables as shared variables so model can be run on GPU

        #encoder
        W_xhe = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder,features)).astype(theano.config.floatX), name='W_xhe')
        W_hhe = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder,hidden_units_encoder)).astype(theano.config.floatX), name='W_hhe')
        
        b_he = theano.shared(np.zeros((hidden_units_encoder,1)).astype(theano.config.floatX), name='b_hhe', broadcastable=(False,True))

        W_hmu = theano.shared(np.random.normal(0,sigma_init,(latent_variables,hidden_units_encoder)).astype(theano.config.floatX), name='W_hmu')
        b_hmu = theano.shared(np.zeros((latent_variables,1)).astype(theano.config.floatX), name='b_hmu', broadcastable=(False,True))

        W_hsigma = theano.shared(np.random.normal(0,sigma_init,(latent_variables,hidden_units_encoder)).astype(theano.config.floatX), name='W_hsigma')
        b_hsigma = theano.shared(np.zeros((latent_variables,1)).astype(theano.config.floatX), name='b_hsigma', broadcastable=(False,True))

        #decoder
        W_zh = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,latent_variables)).astype(theano.config.floatX), name='W_zh')
        b_zh = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_zh', broadcastable=(False,True))

        W_hhd = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,hidden_units_decoder)).astype(theano.config.floatX), name='W_hhd')
        W_xhd = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,features)).astype(theano.config.floatX), name='W_hxd')
        
        b_hd = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_hxd', broadcastable=(False,True))
        
        W_hx = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_decoder)).astype(theano.config.floatX), name='W_hx')
        b_hx = theano.shared(np.zeros((features,1)).astype(theano.config.floatX), name='b_hx', broadcastable=(False,True))

        self.params = OrderedDict([("W_xhe", W_xhe), ("W_hhe", W_hhe), ("b_he", b_he), ("W_hmu", W_hmu), ("b_hmu", b_hmu), \
            ("W_hsigma", W_hsigma), ("b_hsigma", b_hsigma), ("W_zh", W_zh), ("b_zh", b_zh), ("W_hhd", W_hhd), ("W_xhd", W_xhd), ("b_hd", b_hd),
            ("W_hx", W_hx), ("b_hx", b_hx)])

        #Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key,value in self.params.items():
            if 'b' in key:
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key, broadcastable=(False,True))
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key, broadcastable=(False,True))
            else:
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)


    def create_gradientfunctions(self,data):
        """This function takes as input the whole dataset and creates the entire model"""
        def encodingstep(x_t, h_t):
            return T.tanh(self.params["W_xhe"].dot(x_t) + self.params["W_hhe"].dot(h_t) + self.params["b_he"])

        x = T.tensor3("x")

        h0_enc = T.matrix("h0_enc")
        result, _ = theano.scan(encodingstep, 
                sequences = reverse_x, 
                outputs_info = h0_enc)

        h_encoder = result[-1]

        #log sigma encoder is squared
        mu_encoder = T.dot(self.params["W_hmu"],h_encoder) + self.params["b_hmu"]
        log_sigma_encoder = T.dot(self.params["W_hsigma"],h_encoder) + self.params["b_hsigma"]

        #Use a very wide prior to make it possible to learn something with Z
        logpz = 0.005 * T.sum(1 + log_sigma_encoder - mu_encoder**2 - T.exp(log_sigma_encoder), axis = 0)

        seed = 42
        
        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
            srng = T.shared_randomstreams.RandomStreams(seed=seed)

        #Reparametrize Z
        eps = srng.normal((self.latent_variables,self.batch_size), avg = 0.0, std = 1.0, dtype=theano.config.floatX)
        z = mu_encoder + T.exp(0.5 * log_sigma_encoder) * eps

        h0_dec = T.tanh(self.params["W_zh"].dot(z) + self.params["b_zh"])

        def decodingstep(x_t, h_t):
            h = T.tanh(self.params["W_hhd"].dot(h_t) + self.params["W_xhd"].dot(x_t) + self.params["b_hd"])
            x = T.nnet.sigmoid(self.params["W_hx"].dot(h) + self.params["b_hx"])

            return x, h

        x0 = T.matrix("x0")
        [y, _], _ = theano.scan(decodingstep,
                n_steps = x.shape[0], 
                outputs_info = [x0, h0_dec])

        # Clip y to avoid NaNs, necessary when lowerbound goes to 0
        y = T.clip(y, 1e-6, 1 - 1e-6)
        logpxz = T.sum(-T.nnet.binary_crossentropy(y,x), axis = 1)

        logpxz = T.mean(logpxz, axis = 0)

        #Average over time dimension
        logpx = T.mean(logpxz + logpz)

        #Compute all the gradients
        gradients = T.grad(logpx, self.params.values())

        #Let Theano handle the updates on parameters for speed
        updates = OrderedDict()
        epoch = T.iscalar("epoch")
        gamma = T.sqrt(1 - (1 - self.b2)**epoch)/(1 - (1 - self.b1)**epoch)

        #Adam
        for parameter, gradient, m, v in zip(self.params.values(), gradients, self.m.values(), self.v.values()):
            new_m = self.b1 * gradient + (1 - self.b1) * m
            new_v = self.b2 * (gradient**2) + (1 - self.b2) * v

            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v)+ 1e-8)
            updates[m] = new_m
            updates[v] = new_v

        batch = T.iscalar('batch')

        givens = {
            h0_enc: np.zeros((self.hidden_units_encoder,self.batch_size)).astype(theano.config.floatX), 
            x0:     np.zeros((self.features,self.batch_size)).astype(theano.config.floatX),
            x:      data[:,:,batch*self.batch_size:(batch+1)*self.batch_size]
        }

        self.updatefunction = theano.function([batch,epoch], logpx, updates=updates, givens=givens, allow_input_downcast=True)

        return True

    def encode(self, x):
        """Helper function to compute the encoding of a datapoint to z"""
        h = np.zeros((self.hidden_units_encoder,1))

        W_xhe = self.params["W_xhe"].get_value()
        b_xhe = self.params["b_xhe"].get_value()
        W_hhe = self.params["W_hhe"].get_value()
        b_hhe = self.params["b_hhe"].get_value()
        W_hmu = self.params["W_hmu"].get_value()
        b_hmu = self.params["b_hmu"].get_value()
        W_hsigma = self.params["W_hsigma"].get_value()
        b_hsigma = self.params["b_hsigma"].get_value()

        for t in xrange(x.shape[0]):
            h = np.tanh(W_xhe.dot(x[t,:,np.newaxis]) + b_xhe + W_hhe.dot(h) + b_hhe)

        mu_encoder = W_hmu.dot(h) + b_hmu
        log_sigma_encoder = W_hsigma.dot(h) + b_hsigma

        z = np.random.normal(mu_encoder,np.exp(log_sigma_encoder))

        return z, mu_encoder, log_sigma_encoder

    def decode(self, t_steps, latent_variables, z = None):
        """Helper function to compute the decoding of a datapoint from z to x"""
        if z == None:
            z = np.zeros((latent_variables,1))

        x = np.zeros((t_steps+1,self.features))

        W_zh = self.params['W_zh'].get_value()
        b_zh = self.params['b_zh'].get_value()

        W_hhd = self.params['W_hhd'].get_value()
        b_hhd = self.params['b_hhd'].get_value()

        W_xhd = self.params['W_xhd'].get_value()
        b_xhd = self.params['b_xhd'].get_value()

        W_hx = self.params['W_hx'].get_value()
        b_hx = self.params['b_hx'].get_value()

        h = W_zh.dot(z) + b_zh

        for t in xrange(t_steps):
            h = np.tanh(W_hhd.dot(h) + b_hhd + W_xhd.dot(x[t,:,np.newaxis]) + b_xhd)
            x[t+1,:] = np.squeeze(1 /(1 + np.exp(-(W_hx.dot(h) + b_hx))))
        
        return x[1:,:]

    def save_parameters(self, path):
        """Saves all the parameters in a way they can be retrieved later"""
        pickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/params.pkl", "wb"))
        pickle.dump({name: m.get_value() for name, m in self.m.items()}, open(path + "/m.pkl", "wb"))
        pickle.dump({name: v.get_value() for name, v in self.v.items()}, open(path + "/v.pkl", "wb"))

    def load_parameters(self, path):
        """Load the variables in a shared variable safe way"""
        p_list = pickle.load(open(path + "/params.pkl", "rb"))
        m_list = cPickle.load(open(path + "/m.pkl", "rb"))
        v_list = cPickle.load(open(path + "/v.pkl", "rb"))

        for name in p_list.keys():
            self.p[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))

