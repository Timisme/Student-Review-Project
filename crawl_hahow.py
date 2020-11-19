import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
import requests

url = 'https://api.hahow.in/api/courses/5ebca40b454a0417c5880c8e/feedbacks?limit=20&page=0'

text = requests.get(url).text

print(text)