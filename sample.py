# -*- coding: utf-8 -*-
import data_helpers
import keras
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

x = data_helpers.my_get_input_sentence()
model = keras.models.load_model('./simple_net.h5')
y = model.predict_on_batch(x)


result = model.predict_proba(x)


print(round(y), float(result))
