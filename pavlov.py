import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov import train_model
from deeppavlov.core.commands.infer import build_model

model_config = read_json('en_ranker_tfidf_data.json')
ranker = train_model(model_config)

print('==========RANKER======', ranker(['accidents']))



# Download all the SQuAD models
# squad = build_model(configs.squad.multi_squad_noans_infer, download = True)

# Do not download the ODQA models, we've just trained it
odqa = build_model('en_odqa_infer_data.json', download = False)

val_q = "What causes accidents?"
answer1 = odqa([val_q]) #  provide answer based on trained data 

print(answer1)
