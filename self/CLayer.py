from keras import backend as K
from keras.models import Model
from keras.layers import Layer,Input

class TripleLayer(Layer):
    def __init__(self,alpha,**kwargs):
        self.alpha = alpha
        super(TripleLayer,self).__init__(**kwargs)

    def tripleLoss(self,inputs):
        a,p,n = inputs
        p_dist = K.sum(K.square(a-p),axis= -1 )
        n_dist = K.sum(k.square(a-n),axis=-1)
        return K.sum(K.maximum(p_dist-n_dist+self.alpha,0),axis = 0)
    def call(self,inputs,**kwargs):
        loss = self.tripleLoss(inputs)
        self.add_loss(loss)

in_a = Input(shape=(96,96,3))
in_p = Input(shape=(96,96,3))
in_c = Input(shape=(96,96,3))


emb_a = nn4_small2(in_a)
emb_p = nn4_small2(in_p)
emb_n = nn4_small2(in_n)

triple_loss_layer = TripleLayer(alpha=0.2,name='triple_loss_layer')([emb_a, emb_p, emb_n])
