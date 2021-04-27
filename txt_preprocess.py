import pandas as pd
import numpy as np 
import re
from langdetect import detect

class preprocess():
	def __init__(self, df):

		for idx, txt in enumerate(df['文本']):
			if type(txt) != str:
				df.drop(320, inplace= True)

		for idx, txt in zip(df.index,df['文本']):
			try:
				if detect(txt) == 'en':
					df.drop(idx, inplace= True)
				else: 
					df.loc[idx, 'lang'] = detect(txt)
			except:
				pass
		df['文本'] = df['文本'].apply(lambda txt: self.preprocess_txt(txt))
		df = df[df['文本'].apply(self.text_len)>4].copy()
		df.drop_duplicates(subset=['文本'],inplace= True)
		df_new = df.reset_index().rename(columns={"index": "文本編號"})
		df_new['文本'] = df_new['文本'].apply(self.split_txt)

		data = []
		for idx, row in df_new.iterrows():
			for sub_txt in row['文本']:
				data.append([row['文本編號'], sub_txt, row['標記'], 	row['註記'], row['lang']])
		cols = df_new.columns
		df_cut = pd.DataFrame(data,columns = cols)
		df_cut = df_cut[df_cut['文本'].apply(self.text_len)>4].drop(['lang'],axis=1)
		
		self.df = df
		self.df_cut = df_cut

		return None

	def preprocess_txt(self, txt):
		txt = re.sub(r'\s+','',txt)
		txt = re.sub(r',', '，', txt) #把, 轉成，
		txt = re.sub(r'[ˇ></)/(/）/（“”「」]','', txt) # 移除ˇ > < ) ( 符號。
		return txt

	def text_len(self, txt):
		return len(txt)

	def split_txt(self, txt):
		results = re.split(r'\d+\.|。|；|!|[?]|？|！', txt) # split txt by [number]. or 。
		return list(filter(None, results))

	# def preprocess(df):

if __name__ == '__main__':

	df = pd.read_csv('data/train_data_label.csv', delimiter=',',encoding='ANSI', names=['文本','標記','註記'])
	df_cut = preprocess(df).df_cut
	print(df_cut)