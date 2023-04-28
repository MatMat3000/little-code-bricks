
import numpy as np

class math_base(object):

    def __init__(self):
        super(math_base, self).__init__()
        pass

    def fast_1D_topk_argsort(self, array, topk):
        topk_idxs           = np.argpartition(array, -topk)[-topk:]
        topk_idxs_sorted    = topk_idxs[np.argsort(array[topk_idxs])]
        return topk_idxs_sorted

    def fast_2D_topk_argsort_axis1(self, array, topk):
        topk_idxs           = np.argpartition(array, -topk, axis=1)[:, -topk:]
        topk_idxs_sorted    = np.take_along_axis(topk_idxs, np.argsort(np.take_along_axis(array, topk_idxs, axis=1)), axis=1)
        return topk_idxs_sorted

    def get_topk_sim_words(self, word, topk, normalized = True):
        if word not in self.embedds_dict: return None, None
        emb_matrix  = self.embedds_df.to_numpy()
        emb_vec     = self.embedds_dict[word]
        sims        = np.dot(emb_matrix, emb_vec)
        if not normalized:
            sims    /= np.linalg.norm(emb_matrix, axis=1)*np.linalg.norm(emb_vec) # ADD THIS IF EMB NOT NORMALIZED (Slower)
        idxs        = self.fast_1D_topk_argsort(sims, topk+1)[::-1][1:] #reverse for top results first
        return self.embedds_df.index[idxs].to_numpy(), sims[idxs]

    def get_random_topk(self, word, topk, normalize):
        rand_index = np.random.randint(topk)
        if word not in self.mem_random_topk:
            topk_w, topk_sim    = self.get_topk_sim_words(word, topk, normalize)
            if topk_w is None: return word # if word not in dict return itself
            self.mem_random_topk[word] = topk_w
            return topk_w[rand_index]
        return self.mem_random_topk[word][rand_index]