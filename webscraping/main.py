from bs4 import BeautifulSoup
import requests

html_text = requests.get('https://www.swinburneonline.edu.au/faqs/')
soup = BeautifulSoup(html_text.text, 'lxml')