import pandas as pd
import numpy as np 
# from wordcloud import WordCloud
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

jieba.set_dictionary('jieba_traditional.txt')

class info_extractor:
    
    def __init__(self, clean_corpus):
        
        self.clean_corpus= clean_corpus
        
        with open('ch_stop_words.txt', 'r', encoding='utf-8') as f:
            self.stop_words = f.read().splitlines()
            
        jieba.load_userdict('user_dict.txt')
        
        self.vectorizer= TfidfVectorizer(max_df= 0.8,
                              min_df= 0.001,
                              ngram_range= (1,1),
                              stop_words= self.stop_words,
                              token_pattern=r"(?u)\b\w+\b")
        
    def keyword_extraction(self, idx, topK=10, pos= True):

        df_neg = self.clean_corpus[self.clean_corpus['標記'] == -1]
        df_pos = self.clean_corpus[self.clean_corpus['標記'] == 1]

        corpus = list(df_neg['文本']) if pos == False else list(df_pos['文本'])
        
        if idx >= len(corpus):
            print('請輸入小於 {} 的數字'.format(len(corpus)))
            return
        
        corpus_cleaned= []
        for txt in corpus:
            words_cut = jieba.cut(txt, cut_all= False)
            corpus_cleaned.append(' '.join(words_cut))

        tfidf_matrix = self.vectorizer.fit_transform(corpus_cleaned)
        tfidf_df = pd.DataFrame(tfidf_matrix.T.toarray(), index= self.vectorizer.get_feature_names())
        
        keywords = tfidf_df[idx][tfidf_df[idx]>0].sort_values(ascending=False)[:topK]
        print('-----------------------------------\n')
        print('文本編號: {}\n'.format(idx))
        print('關鍵字 及 關鍵字分數:\n ')
        for i, (kw, score) in enumerate(keywords.items()):
            print('{}. {} : {:.2f}'.format(i+1, kw, score))
            
        print('\n原回饋內容:\n\n{}'.format(corpus[idx]))
        print('-----------------------------------\n') 
        return list(jieba.cut(corpus[idx]))
    
    def document_matching(self, input_txt, topK = 3):
        
        corpus = list(self.clean_corpus['文本'])

        corpus.append(input_txt)

        corpus_cut= []
        for txt in corpus:
            words_cut = jieba.cut(txt)
            corpus_cut.append(' '.join(words_cut))

        vectorizer= TfidfVectorizer(max_df= 0.9,
                              min_df= 0.001,
                              ngram_range= (1,3),
                              stop_words= None,
                              token_pattern=r"(?u)\b\w+\b")

        tfidf_matrix = vectorizer.fit_transform(corpus_cut)

        cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix).flatten()

        related_docs_indices = cosine_similarities.argsort()[:-(topK+2):-1]
        # print(related_docs_indices)
        # print(cosine_similarities[related_docs_indices])
        print('-----------------------------------\n')
        print('查詢關鍵字: {}\n'.format(txt))
        print('前 {} 個最佳符合的回饋內容:\n'.format(topK))
        for idx, (docu_idx, score) in enumerate(zip(related_docs_indices[1:], cosine_similarities[related_docs_indices[1:]])):
            print('{}. {} 相似程度: {:.2f}\n'.format(idx+1, corpus[docu_idx], score))
        print('-----------------------------------\n')

if __name__ == '__main__':

    df_clean = pd.read_csv('df_clean.csv')
    extractor = info_extractor(df_clean)

    extractor.keyword_extraction(355, pos=True)
    txt= '上課遲到'
    extractor.document_matching(txt)