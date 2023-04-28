
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd, numpy as np, re
import math_utils

class random_topk_word_sampler(math_utils.math_base):

    def __init__(self, embedds_path, vocab_size=15_000, normalize= True, word_change_per_sentence=1, topk_permut=5, nb_sample=1, omit_word_size = 3,  method='random'):

        super(random_topk_word_sampler, self).__init__()

        ### init attributes

        # memoization
        self.mem_random_topk  = {}
        self.mem_imp_words    = {}

        #parameters
        self.vocab_size                 = vocab_size
        self.normalize                  = normalize
        self.word_change_per_sentence   = word_change_per_sentence
        self.topk_permut                = topk_permut
        self.nb_sample                  = nb_sample
        self.omit_word_size             = omit_word_size
        self.method                     = method

        # word embedings
        self.embedds_df     = None              # elem for dot product of similarities
        self.embedds_dict   = None              # elem for quick embedding lookup 
        self.vocabulary     = None

        ### init calls
        self._load_embedds(embedds_path, vocab_size, normalize)

    def _load_embedds(self, path, vocab_size, normalize=True):

        print("loading embeddings ... ", end='')

        with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f) #jump first line
            data = {}
            for i, line in enumerate(f):
                if i==vocab_size: break
                tokens = line.rstrip().split(' ')
                word = tokens[0].lower()
                embedding = tokens[1:]
                if word not in data:
                    data[word] = np.array(embedding, dtype='float32') # [float(x) for x in tokens[1:]] gives big emory leak
                    if normalize: data[word] /= np.linalg.norm(data[word]) # Normalize embedding for optimisation
        
        self.embedds_df     = pd.DataFrame(data).T
        self.embedds_dict   = data
        self.vocabulary     = np.array(list(data))
        
        print("[OK]")

    def remove_short_words(self, text, omit_word_size):
        regexp = r'\b'+omit_word_size*r'\w'+r"\w+[-'’]?\w*\b"
        return ' '.join(re.findall(regexp, text))
    
    def get_sent_imp_words(self, data, word_change_per_sentence):
        ''' return important word to substitute (data.shape[0], nb_imp_words):
            [   ['w11', 'w12', 'w13'],
                ['w21', 'w22', 'w23'],
                    ...         
                ['w11', 'w22', 'w33']   ] '''

        ID = id(data)
        
        if ID not in self.mem_imp_words:

            if self.method=='tf-idf':
                vectorizer      = TfidfVectorizer(vocabulary = self.vocabulary, token_pattern=r"\b\w\w+[-'’]?\w*\b")
                tf_idf_matrix   = vectorizer.fit_transform(data).toarray()
                idxs            = self.fast_2D_topk_argsort_axis1(tf_idf_matrix, word_change_per_sentence+1)[:, ::-1][:, 1:] # reverse to have best first & omit itself
                self.mem_imp_words[id(data)] = pd.DataFrame(self.vocabulary[idxs])

            if self.method == 'random':
                def rand_select(elem):
                    elem = elem.split()
                    if len(elem)<1: return None
                    rand_idxs   = np.random.randint(len(elem), size=word_change_per_sentence)
                    return np.array(elem)[rand_idxs]

                self.mem_imp_words[ID] = pd.DataFrame(data.apply(rand_select).to_list())


        return self.mem_imp_words[ID]

    def permut_word(self, sentence, word, new_word):
        return  sentence.replace(word, new_word)

    def augment_data(self, df, aug_col):

        data = df[aug_col]
        aug_df = pd.DataFrame()        
        aug_df['original'] = data

        data_without_short_word = data.apply(lambda x : self.remove_short_words(x, self.omit_word_size))
        imp_words = self.get_sent_imp_words(data_without_short_word, self.word_change_per_sentence)

        for i in range(self.nb_sample):
            aux = data.copy()
            for ii in range(self.word_change_per_sentence):
                aux = list(map(self.permut_word, aux, imp_words[ii], imp_words[ii].apply(lambda x : self.get_random_topk(x, self.topk_permut, self.normalize))))
        
            aug_df[f'sample_{i}'] = aux
        
        self.transform_view = aug_df
        stacked_results = aug_df.stack().reset_index(level=1, drop=True).rename('aug_data')

            
        return pd.concat([df.drop(aug_col, axis=1), stacked_results], axis=1)
