import requests
from bs4 import BeautifulSoup
import os
import sys


#possibly useless

PATH= os.path.dirname(__file__)

download_path = os.path.join(PATH,'testPDF')

import re

from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
import requests, lxml, os, json


import requests
from bs4 import BeautifulSoup
import json

def scrape_one_google_scholar_page(term, DEBUG=False):
    headers = {
        'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
    }
    params = {
        'q': term,
        'hl': 'en'
    }
    try:
        html = requests.get('https://scholar.google.com/scholar', headers=headers, params=params).text
        if 'Our systems have detected unusual traffic from your computer network.' in html:
            return -2
        
        soup = BeautifulSoup(html, 'lxml')
        data = []

        # Process only the first result

        result = soup.select_one('.gs_r.gs_or.gs_scl')

        if result:
            
            title = result.select_one('.gs_rt').text
            title_link = result.select_one('.gs_rt a')['href']
            publication_info = result.select_one('.gs_a').text
            snippet = result.select_one('.gs_rs').text
            cited_by = result.select_one('#gs_res_ccl_mid .gs_nph+ a')['href']

            try:
                pdf_link = result.select_one('.gs_or_ggsm a:nth-child(1)')['href']
            except:
                pdf_link = None

            data.append({
                'title': title,
                'title_link': title_link,
                'publication_info': publication_info,
                'snippet': snippet,
                'cited_by': f'https://scholar.google.com{cited_by}',
                'pdf_link': pdf_link
            })

            # Extract the citation count from the "cited by" link
            cited_by_count = 0
            cited_by_count_match = soup.find('a', {'href': cited_by})
            if cited_by_count_match:
                # Extract the citation count from the tag's text
                cited_by_count = int(cited_by_count_match.text.strip().split()[-1])

            if DEBUG:
                print(json.dumps(data))
        else:
            cited_by_count = -1
    except Exception as e:
        print(e)
        cited_by_count = -1

    return cited_by_count



if __name__ == '__main__':
    citations=scrape_one_google_scholar_page("tony",DEBUG=False)
    print(citations)