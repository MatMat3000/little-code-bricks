# French word data augmenter

Simple word augmenter in french. It swaps a few words with other words close in meaning (embedding wise using cosine sim)

- Uses pre trained small french Fasttext embeddings from META AI (100k_cc.fr.300.vec file, need unzipping)
- Uses Memoization for faster computations
- Chooses which word are to be swapped (using tf-idf or randomly)
- Many paremeters to tune (nb of words per sentence, top k most similar words, min word length to be considered a word)

# Test demo usage

   ```sh
   python nlp_french_data_aug
   ```
