from keras.layers import Layer
import keras.backend as K

class CRF(Layer):
    """
    CRF层时一个带训练参数的loss计算层 crf只用来训练模型 预测需要另外建立模型
    苏剑林. (2018, May 18). 《简明条件随机场CRF介绍（附带纯Keras实现） 》[Blog post]. Retrieved from https://kexue.fm/archives/5542
    """
    def __init__(self,ignore_last_label=False,**kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF,self).__init__(**kwargs)
    def build(self,input_shape):
        self.num_labels = input_shape[-1] -self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
        shape=(self.num_labels,self.num_labels),
        initializer='glorot_uniform',
        trainable=True)
    def log_norm_step(self,inputs,states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        states = K.expand_dims(states[0],2)# (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans,0)#(1.output_dim,output_dim)
        output = K.logsumexp(states+trans,1)#(batch_size,output_dim)
        return output+inputs,[output+inputs]
    def path_score(self,inputs,labels):
        """计算目标路径的相对概率(未归一化)
        逐标签得分，加上转移概率得分
        用预测点乘目标的方法抽取目标路径得分
        """
        point_scoe = K.sum(K.sum(inputs*labels,2),1,keepdims=True)#逐标签得分
        labels1 = K.expand_dims(labels[:,:-1],3)
        labels2 = K.expand_dims(labels[:,1:],2)
        labels = labels1 * labels2
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans*labels, [2,3]), 1, keepdims=True)
        return point_scoe+trans_score
    def call(self,inputs):#crf不改变输出 loss函数
        return inputs
    def loss(self,y_true,y_pred):#目标y_pred是one hot形式
        mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_states = [y_pred[:,0]] # 初始状态
        log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        return log_norm - path_score # 即log(分子/分母)
    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)