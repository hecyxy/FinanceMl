import bilstm_crf
EPOCHS = 4
model,(train_x,train_y),(text_x,test_y) = bilstm_crf.create_model()
model.fit(train_x,train_y,batch_size=500,epochs=EPOCHS,validation_data=[text_x,test_y])
model.save('ner/model/search0.h5')