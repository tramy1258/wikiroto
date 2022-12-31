#import wikipedia
#from bs4 import BeautifulSoup
#import numpy as np
import time
import requests
import pandas as pd
import re
import json
import sys

##### LOADING DATA ####
rotodf = pd.read_csv('../data/rotten_tomatoes_critic_reviews.csv')
criticdf = rotodf.groupby('critic_name').agg(lambda x: list(x)).reset_index('critic_name')
print(len(criticdf),'users')

#### SCRAPING FUNCTIONS ####
headers = {
    'Referer': 'https://www.rottentomatoes.com/m/notebook/reviews?type=user',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}

s = requests.Session()
        
def get_reviews(critic_link):

    api_url = f"https://www.rottentomatoes.com/napi/critics/{critic_link}/movies"
    
    payload = {
        'direction': 'next',
        'endCursor': '',
        'startCursor': '',
    }
    
    review_data = []
    i = 0
    stcode = 0
    with requests.Session() as s:
        while i < 50:
            try:
                r = s.get(api_url, headers=headers, params=payload)
            except:
                print('problem with request',end=' ')
                break
            i += 1
            stcode = r.status_code
            if r.status_code != 200:
                print(r.reason,r.status_code,end=' ')
                break

            try:
                data = r.json()
            except json.decoder.JSONDecodeError:
                print('json error??',end=' ')
                break
            except:
                print('weird error??',end=' ')
                break

            quotes = [q['quote'] for q in data['reviews']]
            review_data.extend(quotes)

            if not data['pageInfo']['hasNextPage']:
                break

            payload['endCursor'] = data['pageInfo']['endCursor']
            payload['startCursor'] = data['pageInfo']['startCursor'] if data['pageInfo'].get('startCursor') else ''

            time.sleep(1)

        #for r in review_data:
        #    print('-->',r)
    
    return review_data, stcode, i

non_eng = {
            'ñ':'n','ç':'c',
            'ü':'u','ú':'u','ù':'u',
            'ä':'a','á':'a','à':'a',
            'ë':'e','é':'e','í':'i',
            'ô':'o','ó':'o','ö':'o'
            }

def get_critic_link(n):
    
    name = n['critic_name'].lower()
    name = re.sub('[\'.!,"#;]', '', name)
    name = re.sub('[&]','and',name)
    if '(' in name:
        name = name[:name.index('(')]
    name = re.sub('[ ]','-',name.strip(' '))

    if not name.isascii():
        #print(i,name,end=', ')
        for c in name:
            if c in non_eng:
                name = name.replace(c,non_eng[c])
        #print(name)
        
    return name


#### SCRAPING ####
resjson = {'entries':[]}
start_ind = int(sys.argv[1]) #int(sys.argv[1])*1000
end_ind = (start_ind//3000)*3000+3000
print(f'scraping {start_ind} to {end_ind-1}')

totalreq = 0
for i,n in criticdf[start_ind:end_ind].iterrows():
    print(f"--{i}--  {n['critic_name']} --> ",end='')
    link = get_critic_link(n)
    print(link, end=' --> ')
    reviews,code,nbreq = get_reviews(link)
    totalreq += nbreq
    print(len(reviews))
    resjson['entries'].append({
                                n['critic_name']:{
                                                'critic_link':link,
                                                'critic_reviews':reviews
                                                }
                                })
    if (i+1) % 10 == 0:
        with open(f'../data/roto_all_critic_reviews_{(i+1)//1000}.json','w') as f:
            print(f'writing to ../data/roto_all_critic_reviews_{(i+1)//1000}.json')
            json.dump(resjson, f, indent=4, ensure_ascii=False)
            
    if not (code==200 or code==404):
        break

print('DONE SCRAPING',totalreq)

#### FINAL SAVING TO JSON FILE ####
with open(f'../data/roto_final_all_critic_reviews_{end_ind-3000}.json','w') as f:
    json.dump(resjson, f, indent=4, ensure_ascii=False)

#### CHECKING SAVED FILE ####
with open(f'../data/roto_final_all_critic_reviews_{end_ind-3000}.json','r') as f:    
    data = json.load(f)

name = list(data['entries'][0].keys())[0]
print(len(data['entries'][0][name]['critic_reviews']),data['entries'][0][name]['critic_reviews'][0])
#print(data['entries'][0]['critic_reviews'][0])