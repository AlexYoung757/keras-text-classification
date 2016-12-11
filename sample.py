# -*- coding: utf-8 -*-
import data_helpers
import keras
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

x = data_helpers.my_get_input_sentence()
model = keras.models.load_model('./save.h5')
y = model.predict(x)

result = model.predict_proba(x)

print(x)
print(y)
print(round(y), float(result))
