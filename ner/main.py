import bilstm_crf
import process_data
import numpy as np
#此文件下搜索数据量较少 测试效果是不佳的
# 'O','B-LOC', 'I-LOC', "B-ORG", "I-ORG","B-PRO","I-PRO"]
# O 代表other
# B-PRO  职位
# B-ORG  公司
# B-LOC 地点
model, (vocab, chunk_tags) = bilstm_crf.create_model(train=False)
predict_text = '伊诺尔信息技术高级产品经理'#'中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
strs, length = process_data.process_data(predict_text, vocab)
print(len(strs))
model.load_weights('ner/model/search2.h5')
raw = model.predict(strs)
print(raw.shape)
result = [np.argmax(row) for row in raw[0][-length:]]
result_tags = [chunk_tags[i] for i in result]

org, loc, pro = '', '', ''

for s, t in zip(predict_text, result_tags):
    if t in ('B-ORG', 'I-ORG'):
        org += ' ' + s if (t == 'B-ORG') else s
    if t in ('B-PRO','I-PRO'):
        pro += ' ' + s if( t == 'B-PRO') else s

print(['organzation:' + org,'PROJECT:' + pro])