from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov import configs, train_model

model_config = read_json(configs.doc_retrieval.en_ranker_tfidf_wiki)
model_config["dataset_reader"]["data_path"] = "data"
model_config["dataset_reader"]["dataset_format"] = "txt"
ranker = train_model(model_config)

from deeppavlov.core.commands.infer import build_model

# Download all the SQuAD models
squad = build_model(configs.squad.multi_squad_noans_infer, download = True)

# Do not download the ODQA models, we've just trained it
odqa = build_model(configs.odqa.en_odqa_infer_wiki, download = False)

val_q = "What causes accidents?"
answer1 = odqa([val_q]) #  provide answer based on trained data 

print(answer1)