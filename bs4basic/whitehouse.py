from cgitb import text
import requests
from bs4 import BeautifulSoup

result = requests.get("https://www.whitehouse.gov/briefings-statements/")
src = result.content
soup = BeautifulSoup(src, 'lxml')

urls = []
for h2_tag in soup.find_all('h2'):
    # a_tag=[]
    # a_tag = h2_tag.find('a')
    urls.append(h2_tag.text)

print(urls)