import numpy as np
import math
import random

SAMPLE_RATE = 44100
CLIP_DURATION = 1
N_SAMPLES = SAMPLE_RATE*CLIP_DURATION
MAX_RECURSION = 2


def generate_envelope(a, d, s, r, n_samples, sample_rate):
    a_samp = math.floor(a*sample_rate)
    d_samp = math.floor(d*sample_rate)
    r_samp = math.floor(r*sample_rate)

    out = np.hstack([np.linspace(0, 1, a_samp), np.linspace(
        1, s, d_samp), np.linspace(s, 0, r_samp), np.zeros((n_samples-a_samp-d_samp-r_samp))])

    return out


class Operator:
    def __init__(self, base_freq, adsr, amp):
        self.modulators = []
        self.base_freq = base_freq
        self.adsr = adsr
        self.amp = amp

    def generate(self, rl=0):
        if rl > MAX_RECURSION:
            return np.zeros((N_SAMPLES))
        freq = np.ones((N_SAMPLES))*self.base_freq
        modulated_freq = freq*0
        for m in self.modulators:
            modulated_freq = modulated_freq+freq*m.generate(rl+1)
        phase_in = np.cumsum((freq+modulated_freq)/SAMPLE_RATE)
        tone = np.sin(2*math.pi*phase_in)
        env = generate_envelope(
            self.adsr[0], self.adsr[1], self.adsr[2], self.adsr[3], N_SAMPLES, SAMPLE_RATE)
        return tone*env*self.amp


N_OPS = 5
N_PARAMS = 6


class SimpleFM:

    def denormalize_freq(self, x):
        return 20+2000*x

    def denormalize_dur(self, x):
        return x*0.3

    def get_op_params(self,):
        return [random.random() for i in range(6)]

    def denormalize_op_params(self, X):
        return [self.denormalize_freq(X[0]), self.denormalize_dur(X[1]), self.denormalize_dur(X[2]), X[3], self.denormalize_dur(X[4]), X[5]]

    def generate_params(self):
        folded_op_params = [self.get_op_params() for i in range(5)]
        return [item for sublist in folded_op_params for item in sublist]

    def denormalize_params(self, params):
        folded_op_params = [
            params[i*N_PARAMS:(i+1)*N_PARAMS] for i in range(N_OPS)]
        folded_dn_params = [self.denormalize_op_params(
            pr) for pr in folded_op_params]
        return folded_dn_params

    def generate(self, params):
        folded_dn_params = self.denormalize_params(params)
        ops = [Operator(pr[0], [pr[1], pr[2], pr[3], pr[4]], pr[5])
               for pr in folded_dn_params]
        ops[0].modulators = [ops[1], ops[2]]
        ops[1].modulators = [ops[3], ops[4]]
        ops[2].modulators = [ops[2]]
        w2 = ops[0].generate()
        return w2
