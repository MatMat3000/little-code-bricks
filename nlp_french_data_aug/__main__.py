import pandas as pd
import word_sampler

txt_data = pd.DataFrame(["Le chat saute au dessus du chien qui court dans la foret.","Une souris verte qui courrait dans les champs.",], columns=['texte'])

tf_idf_ws = word_sampler.random_topk_word_sampler('nlp_french_data_aug/100k_cc.fr.300.vec',
                                                  vocab_size=10_000,
                                                  normalize= True, # faster sim calculations
                                                  word_change_per_sentence=2,
                                                  topk_permut=5, 
                                                  nb_sample=2,
                                                  omit_word_size=3,
                                                  method='random')

print(tf_idf_ws.augment_data(txt_data, 'texte'))
print(tf_idf_ws.transform_view)