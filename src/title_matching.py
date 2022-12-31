from bs4 import BeautifulSoup
import bs4
import requests
import re
import pandas as pd
import wikipedia as wp
import csv
import random
import os
from tqdm import tqdm
from numpy import nan
from langdetect import detect, DetectorFactory
import sys
import json

wiki_url = 'https://en.wikipedia.org/wiki/'
roto_url = 'https://www.rottentomatoes.com/'
lower_words = ['The','From','On','In','A','Of','By','With','For','To','An','And']

DetectorFactory.seed = 0

##### to get all normal wiki titles #####
def try_get_title(title,title_dict):
    '''Try to get wikipedia page of name 'title', its summary and its url'''
    titles = dict()
    print(f'===> trying {title}')
    try:
        p = wp.WikipediaPage(title = title)
        if detect(p.summary) == 'en':
            titles[title] = (p.summary,p.url)
    except Exception as e:
        print(e)
    title_dict.update(titles)


def multilang_titles(title):
    '''Seperate title of form 'Title in one language (Title in another language)' into 2 titles'''
    if '(' in title and ')' in title:
        first_bracket = title.index('(')
        last_bracket = title.index(')')
        return [title[:first_bracket-1], title[first_bracket+1:last_bracket]]
    elif ' - ' in title:
        hyphen = title.index(' - ')
        return [title[:hyphen],title[hyphen+3:]]
    else:
        return []


def get_real_wiki_title(title,date,title_dict):
    '''Try all possible versions of 1 title'''

    # trying 'movie_title_(film)'
    new_title = title + ' (film)'
    try_get_title(new_title,title_dict)

    # trying 'movie_title_(year_film)'
    if len(title_dict) == 0:
        if date is not nan: # if release date is available
            year = date[:4]
            new_title = title + f' ({year} film)'
            try_get_title(new_title,title_dict)
            if len(title_dict) == 0:
                new_title = title + f' ({int(year)-1} film)'
                try_get_title(new_title,title_dict)

                new_title = title + f' ({int(year)+1} film)'
                try_get_title(new_title,title_dict)

    # trying 'movie_title'
    if len(title_dict) == 0:
        try_get_title(title,title_dict)
    

def all_titles(t):
    '''Filter resulting titles'''
    titles = dict()
    pos_titles = [t['movie_title']]  #t['movie_title'].replace(' The ',' the '),
    pos_titles += multilang_titles(t['movie_title'])
    #pos_titles = [title.replace(' ','_') for title in pos_titles]
    for orig_title in pos_titles:
        get_real_wiki_title(orig_title,t['original_release_date'],titles)
        ## titles now contains at most 2 titles (original,film), (original,year) or (year-1,year+1)
            
        if len(titles) > 0:
            ### a page with 'movie_title' alone (without extra information) is less likely to be about a film
            #if orig_title in titles and len(titles) > 1:
            #    titles.pop(orig_title)

            ### titles now consist of one film/year_title or 2 year_titles
            ### if both 'title_(year-1)' and 'title_(year+1)' exist, compare their summaries
            if len(titles) > 1:
                summaries = list(titles.values())
                if summaries[0][0] == summaries[1][0]:
                    titles.popitem()
            return titles # contains only 1 title or 2 titles with different summaries

    return titles


##### to get other versions of title if above failed #######
def correct_title(title):
    for w in lower_words:
        title = title.replace(f' {w} ',f' {w.lower()} ').replace(f': {w.lower()} ',f': {w} ').replace('&','and')
    return title

def all_weird_titles(t):
    '''Filter resulting titles'''
    titles = dict()
    if t['wiki_title'] is not nan:
        return None
    pos_titles = [correct_title(t['movie_title'])]
    pos_titles += multilang_titles(t['movie_title'])
    #pos_titles = [title.replace(' ','_') for title in pos_titles]
    for orig_title in pos_titles:
        get_real_wiki_title(orig_title,t['original_release_date'],titles)
        ## titles now contains at most 2 titles (original,film), (original,year) or (year-1,year+1)
            
        if len(titles) > 0:
            ### a page with 'movie_title' alone (without extra information) is less likely to be about a film
            #if orig_title in titles and len(titles) > 1:
            #    titles.pop(orig_title)

            ### titles now consist of one film/year_title or 2 year_titles
            ### if both 'title_(year-1)' and 'title_(year+1)' exist, compare their summaries
            if len(titles) > 1:
                summaries = list(titles.values())
                if summaries[0] == summaries[1]:
                    titles.popitem()
            return titles # contains only 1 title or 2 titles with different summaries

    return titles

##### to get all wiki title #####
def add_wiki_infos(df):
    '''Get titles of whole dataframe while showing progress in terminal'''
    nopos = []
    multipos = []
    ct = 0
    for i,t in tqdm(df.iterrows()):
        real_i = i + int(sys.argv[1])
        if t['wiki_title'] is not nan:
            print(i,'OK',end=',')
            continue
        print('\n===================',i,'===================')
        titsum = all_weird_titles(t)
        if len(titsum) == 1:
            df.at[i,'wiki_title'] = list(titsum.keys())[0]
            df.at[i,'wiki_summary'] = list(titsum.values())[0][0]
            df.at[i,'wiki_url'] = list(titsum.values())[0][1]
            print('------------------->',real_i,'OK')
        elif len(titsum) == 0:
            nopos.append((real_i,t['movie_title'],t['original_release_date']))
            print('------------------->',real_i,'has',len(titsum),'possibilities')
            ct += 1
        else:
            multipos.append((real_i,t['movie_title'],t['original_release_date']))
            print('------------------->',real_i,'has',len(titsum),'possibilities')
            ct += 1
    return ct,nopos,multipos


def to_files(df,nmfile,csvfile):
    '''Run on whole dataframe and save results to files'''
    nmfile = open(nmfile,'w')
    ct,nopos,multipos = add_wiki_infos(df)

    nmfile.write(f'{ct} problems in general\nno match at\n')
    for film in nopos:
        nmfile.write(f'{film}\n')
    nmfile.write(f'multi matches at\n')
    for film in multipos:
        nmfile.write(f'{film}\n')
    nmfile.close()

    df.to_csv(csvfile,index=False)

###### to get all wiki html #########
def get_all_wiki_html(df):
    '''Get whole html from extracted url for the total dataframe'''

    bad_urls = [] # ids where available url does not give a response

    for i,t in df.iterrows():
        #check if html already exists
        if t['wiki_html'] is not nan and t['wiki_html'] != '':
            if i % 100 == 0 or i == len(df)-1:
                df.to_csv(f'../res/wikiroto{sys.argv[3]}.csv',index=False)
            continue

        print(f'==== {i} ====',end='\t')
        #check if url available
        if t['wiki_url'] is not nan:
            print('got URL',end='\t')
            response = requests.get(t['wiki_url'])
            #check if success getting response
            if response.status_code == 200:
                print('got html')
                df.at[i,'wiki_html'] = response.text
            else:
                print('bad URL')
                bad_urls.append(i)
        else:
            print('no URL #######')

        #save to file every 50 rows
        if i % 50 == 0 or i == len(df)-1:
            df.to_csv(f'../res/wikiroto{sys.argv[3]}.csv',index=False)

    return bad_urls


###### to clean infobox #########
def check_good_match(dic,t,i):
    '''Check if the Wiki page found is the one mentioned in Roto data by comparing directors, actors, authors'''
    # List where information about directors, actors, authors not available in Roto but good match
    oklistnan = [212, 1657, 2588, 3010, 3476, 
                4008, 5242, 5367, 5745, 6119, 
                6134, 6212, 6657, 6826, 7021, 
                7795, 8643, 9587, 9589, 10149, 
                10377, 11025, 11408, 12339, 
                12727, 12981, 13744, 13986, 
                15482, 15591, 16424, 16726] 
    if i in oklistnan:
        return True
    
    ok = False
    checklist = ['directors','authors','actors']
    for cat in checklist:
        if t[cat] is not nan:
            # check if at least one name in Wiki page is mentioned in the same category in Roto
            ok = any([name in t[cat] for name in dic.get(cat,[])])
            if ok:
                break
    return ok

def correct_key(key):
    ''' Parse key '''
    key = key.lower()
    if 'direct' in key:
        return 'directors'
    if 'starring' in key:
        return 'actors'
    if 'written' in key or 'screenplay' in key:
        return 'authors'
    if 'produc' in key and 'company' in key:
        return 'production_companies'
    if 'release' in key and 'date' in key:
        return 'release_date'
    return key.replace(' ','_')

def none_to_empty_str(items):
    return {k: v if v is not None else '' for k, v in items}

def remove_bad_match(df,i):
    ''' Remove wrong information when badly matched'''
    df.at[i,'wiki_title'] = nan
    df.at[i,'wiki_summary'] = nan
    df.at[i,'wiki_url'] = nan
    df.at[i,'wiki_html'] = nan

def get_wiki_infobox_dict(df,jsonfile):
    ''' Clean all wiki infobox from html and save good ones to files'''

    jsres = {'entries':[]}
    
    #columns to add to final table, besides wiki table
    col_to_dic = ['movie_title','content_rating','genres','movie_info','critics_consensus','wiki_title','wiki_summary']

    for i,t in df.iterrows():
        if t['wiki_title'] is nan or t['wiki_html'] is nan:
            continue
        print(f"=========={i}========= {t['wiki_title']}",end='\t')
        # dictionary from roto data
        jsdic = json.loads(t[col_to_dic].to_json(),object_pairs_hook=none_to_empty_str)
        # parsing the response
        soup = bs4.BeautifulSoup(t['wiki_html'].replace("<br/>"," ").replace("<br />"," "), 'html.parser')
        # getting infobox
        infobox = soup.find('table', {'class': 'infobox'})
        if infobox is None: #if no infobox available then not a good match
            remove_bad_match(df,i)
            continue        
        #remove all breaks, sup text and span
        for br in infobox.findAll(['br','sup','span'],recursive=True): 
            br.decompose()

        #for each row in table, add an element to jsdic
        for row in infobox.findAll('tr'): 
            key = ''
            for c in row.children:
                if c.name == 'th' and 'infobox-label' in c.get('class',[]):
                    key = correct_key(c.getText())
                if c.name == 'td' and 'infobox-data' in c.get('class',[]):
                    if key != '': #concatenate all data of same label, separated by comma
                        text = ''
                        for l in c.findAll('li'):
                            text += f'{l.getText()}, '
                        if text == '':
                            text = c.getText()
                        text = text.strip(' ,\n\t').replace('\n',', ')
                        if key not in jsdic:
                            jsdic[key] = text
                        else:
                            jsdic[key] += f', {text}'
        #print(json.dumps(jsdic, indent=4, ensure_ascii=False))
        #check if good match with new infobox
        if not check_good_match(jsdic,t,i):
            print('bad match!!!!!!!!!')
            remove_bad_match(df,i)
            continue 
        else:
            print('OK')
        jsres['entries'].append({i:jsdic})
    #write to file    
    with open(jsonfile,'w') as f:
        json.dump(jsres, f, indent=4, ensure_ascii=False)

           
def check_abbrev(s):
    name_titles =  ['Dr','Esq','Hon','Jr',
                'Mr','Mrs','Ms','Messrs',
                'Mmes','Msgr','Prof','Rev',
                'Rt. Hon','Sr','St','lit',
                'translit','a.k.a','vs','No',
                'transl','Inc','Vol'
                ]
    if len(s) <= 7:
        return True
    if s[-1] == '.':
        return True
    if s[-1].isupper() and not s[-2].isalpha():#(s[-2] == ' ' or s[-2] == '.'):
        return True

    for n in name_titles:
        if s[-len(n):] == n and not s[-len(n)-1].isalpha():#(s[-2] == ' ' or s[-2] == '.'):
            return True
    return False

def extract_first_sentence(s):
    print('-'*100)
    ls = s.split('. ')
    #ls = [x.strip(' .') for x in ls]
    ls = [x for x in ls if x != '']
    i = 0
    while i < len(ls)-1:
        print(f'==={i}===>',ls[i],len(ls))
        #wrong separation, concat with next sentence
        if check_abbrev(ls[i]) or ls[i+1][0].islower():
            ls = ls[:i] + [ls[i]+'. '+ls[i+1]] + ls[min(len(ls),i+2):]
        else:
            i +=1
        print(f'     ===>',ls[i],len(ls))
    return ls[0].strip(' .')


############## ROTTEN TOMATOES CRITIC REVIEWS ###################
#roto_review_df = pd.read_csv('../data/rotten_tomatoes_critic_reviews.csv')
#roto_review_user = roto_review_df.groupby('critic_name').agg(lambda x: list(x)).reset_index('critic_name')
#roto_review_film = roto_review_df.groupby('rotten_tomatoes_link').agg(lambda x: list(x)).reset_index('rotten_tomatoes_link')

############### ROTTEN TOMATOES METADATA #####################
#print(f'Reading rows {int(sys.argv[1])} to {int(sys.argv[2])}')
#roto_meta_df = pd.read_csv('../res/wikirotohtml.csv')
#roto_meta_df = pd.read_csv('../data/rotten_tomatoes_movies.csv',header=0,skiprows = lambda x: x in range(1,int(sys.argv[1])+1),nrows=int(sys.argv[2])-int(sys.argv[1]))

################# WIKI FILM INFOBOX AND DESCRIPTION #####################
#roto_meta_df['wiki_title'] = ''
#roto_meta_df['wiki_summary'] = ''
#roto_meta_df['wiki_url'] = ''
#roto_meta_df['wiki_html'] = ''
#print(sum([1 if f is nan else 0 for f in roto_meta_df['wiki_title']]),'films not yet matched out of',len(roto_meta_df),'films.')
#print(sum([1 if f is nan else 0 for f in roto_meta_df['wiki_html']]),'html not yet found out of',len(roto_meta_df),'films.')
#print('Start getting HTML...')
#bad_urls = get_all_wiki_html(roto_meta_df)
#print('Done!')
#print(f'{len(bad_urls)} problems in general at {bad_urls}')
#roto_meta_df.to_csv(f'../res/wikiroto{sys.argv[3]}.csv',index=False)
#to_files(roto_meta_df,f'../res/problemsnew.txt',f'../res/wikirotonew.csv')
#res_roto_meta_df = pd.read_csv(f'../res/wikiroto{sys.argv[3]}.csv')



#get_wiki_infobox_dict(roto_meta_df,'../res/wikiroto.json') 

#extract one sentence summary
weird = []
with open('../res/wikiroto.json') as json_file:
    finaldata = json.load(json_file)
for d in finaldata['entries']:
    print('='*100)
    key = list(d.keys())[0]
    print("###",key,"###",d[key]['wiki_summary'])
    d[key]['wiki_summary'] = extract_first_sentence(d[key]['wiki_summary'])
    if len(d[key]['wiki_summary']) < 35:
        weird.append(key)
    print('---->',d[key]['wiki_summary'])
#save json
with open('../res/wikiroto_onesentence.json','w') as f:
    json.dump(finaldata, f, indent=4, ensure_ascii=False)

print('Done!')
print(weird)