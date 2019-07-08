from yelpapi import YelpAPI
import numpy as np
import pandas as pd
import json
import warnings
from bs4 import BeautifulSoup
import requests
import re
import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import nltk

def getPageUrl(url):
    h = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    page = requests.get(url,headers=h)
    return page

def getSoup(url):
    page = getPageUrl(url)
    soup = BeautifulSoup(page.content,features='lxml')
    return soup

def yelp_df(r_cats,city,outfile,lim=50,total=1000):
	api_key = '1fsQWqAaZCs1VDm4L5KI-cB4nDFpvBAHDTsXIsefm_Ytbw92PRNYJ8Bg3W9ob1fCuUKSLMLki6JQeLktAzhSKsLGM77b-vZEMAk0fJYcy9vKsPLFCmXJOFDRdMJSXHYx'
	yelp_api = YelpAPI(api_key, timeout_s=30.0)

	df = pd.DataFrame(columns=['id','alias','categories','name','image_url','is_closed','url','review_count','rating','transactions','price','phone','display_phone','distance','latitude','longitude','zip'])

	for rc in r_cats:
	    n = 0
	    while n < total:
	        try:
	            search_results = yelp_api.search_query(term=rc,location=city,limit=lim,offset=n)
	            biz = search_results['businesses']

	            for b in biz:
	                categories = b.pop('categories')
	                categories = [i['alias'] for i in categories]
	                b['categories'] = categories

	                coords = b.pop('coordinates')
	                coords = [i for i in coords.values()]
	                b['latitude'] = coords[0]
	                b['longitude'] = coords[1]

	                location = b.pop('location')
	                b['zip'] = location['zip_code']

	            tmp = pd.DataFrame(biz)
	            df = pd.concat([df,tmp],ignore_index=True,axis=0)
	            n+=lim
	        except:
	            print('error')
	            tmp = pd.DataFrame(biz)
	            df = pd.concat([df,tmp],ignore_index=True,axis=0)
	            pass
	    print(rc,end =" ")
	df = df.sort_values('id').drop_duplicates(subset='id')
	df.to_pickle(outfile)


def expand_list(df,lst_col):
	df = pd.DataFrame({
	  col:np.repeat(df[col].values, df[lst_col].str.len())
	  for col in df.columns.drop(lst_col)}
	).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns]
	return df
    
def get_cats(file,type):
	with open(file) as f:
		data = json.load(f)
	r_cats = [d[type] for d in data if len(d['parents'])>0 and d['parents'][0] == 'restaurants']
	return r_cats


def preprocess_restaurants(df,r_cats):
    df = df.drop(columns=['image_url','is_closed','display_phone','phone','transactions'])
    df = expand_list(df,'categories')
    df = df[df['categories'].isin(r_cats)].reset_index(drop=True)
    df = pd.concat([df,pd.get_dummies(df['categories'],prefix='cat',dtype='int64')],axis=1)

    cats = df.columns[df.columns.str.contains('cat_')]
    noncats = df.columns[~df.columns.str.contains('cat_')]

    tmp = df.groupby('alias')[cats].sum().reset_index()
    df = df.drop_duplicates(subset='alias').reset_index(drop=True)
    df = pd.merge(df[noncats],tmp[['alias']+cats.tolist()],on='alias')

    for n in range(1,5):
        df.loc[df['price']=='$'*n,'price']=n
        df.loc[df['price']=='£'*n,'price']=n
        df.loc[df['price']=='€'*n,'price']=n
        df.loc[df['price']=='₱'*n,'price']=n

        
    convert_int = ['review_count']
    convert_float = ['latitude','longitude']
    df[convert_int] = df[convert_int].astype('int64')
    df[convert_float] = df[convert_float].astype('float64',errors='ignore')
    df['price'] = df['price'].fillna(0).astype('int64')

    # df = df.loc[~df['zip'].str.contains(r'[0-9]')]s

    return df


def strip_text(s):
	s = re.sub('[.]',' . ',s)
	s = re.sub('[^a-zA-Z\s]','',s)
	s = " ".join(s.split())
	return s

def get_reviews(url,review_count):
	n = 0
	rpp = 20
	ratings, dates, reviews,locations= [],[],[],[]
	for i in range(0,int(np.floor(review_count/rpp))): 
	    url = '{}?start={}'.format(url.split('?')[0],n)
	    soup = getSoup(url)
	    test = soup.find_all('script', {'type':'application/ld+json'})[-1].text
	    test = re.sub(r"\n", " ", test)
	    test = re.sub(r'\\n',' ',test)

	    ratings.extend(re.compile(r'[{":]*ratingValue[{":]*\s(\d)[}]').findall(test))
	    dates.extend([datetime.datetime.strptime(i,'%Y-%m-%d') for i in re.compile(r'[{":]*datePublished[{":]*\s["]([\d-]*)').findall(test)])
	    r = re.compile(r'[{":]*description[":]*\s(.+?)["]*(author)').findall(test)
	    reviews.extend([strip_text(i[0]) for i in r])
	    loc = soup.find_all('li',class_='user-location responsive-hidden-small')
	    locations.extend([strip_text(i.text) for i in loc])
	    n+=rpp
	    print('.',end="")
	return ratings, dates, reviews, locations

def save_reviews(df,file):
    dfr = pd.DataFrame(columns=['id','dates','ratings','reviews','location'])
    count = 0
    for index,row in df.iterrows():
        if row['review_count']>500:
            ratings,dates,reviews,location = get_reviews(row['url'],row['review_count'])
            dfr = dfr.append(pd.DataFrame({'id': [row['id'] for i in range(0,len(dates))],\
                        'dates':dates,'ratings':ratings,'reviews':reviews,'location':location}),ignore_index=True)
            count+=1
            print(count, end=" ")
    dfr.to_pickle('file')

def return_ts(df_both,cat,level,split='is_local',compare='',state=''):
    
    if compare=='gt':
        df_both = df_both[df_both[cat]>=level]
    elif compare=='lt':
        df_both = df_both[df_both[cat]<=level]
    elif compare=='gtlt':
        df_both = df_both[(df_both[cat]>=level[0]) & (df_both[cat]<level[1])]
    else:
        df_both = df_both[df_both[cat]==level]
    maxdays = df_both.groupby(['id'])['time_diff'].max().max().astype('int64')   
    all_ts = np.zeros([2,df_both['id'].nunique(),maxdays+1])
    all_count = np.zeros([2,df_both['id'].nunique(),maxdays+1])
    for c,i in enumerate(df_both['id'].unique()):
        for l in df_both[split].unique():
            if l==0 and len(state)>0:
                test = df_both.loc[(df_both['state']==state) & (df_both['id']==i) & (df_both[split]==l),['dates','ratings']]
            else:
                test = df_both.loc[(df_both['id']==i) & (df_both[split]==l),['dates','ratings']]
            
            tslen = (test['dates'].max()-test['dates'].min()).days+1
            # get number of reviews
            test2 = test.groupby('dates').size().reset_index()
            test2['dates'] = (test2['dates'].max()-test2['dates']).dt.days
            # test2['dates'] = (test2['dates']-test2['dates'].min()).dt.days
            new = pd.DataFrame({'index':range(tslen),'vals':np.zeros(tslen)})
            new.iloc[test2['dates'].values,1] = test2[0].values
            all_count[l,c,0:tslen] = new['vals'].rolling(60).sum().iloc[::-1]
            all_count[l,c,tslen:] = np.nan
            # get exponentially weighted review
            test = test.groupby('dates')['ratings'].mean().reset_index()
            test['dates'] = (test['dates'].max()-test['dates']).dt.days
            # test['dates'] = (test['dates']-test['dates'].iloc[0]).dt.days
            new = pd.DataFrame({'index':range(tslen),'vals':np.zeros(tslen)})
            new.iloc[test['dates'].values,1] = test['ratings'].ewm(alpha=.1).mean().values
            new[new['vals']==0]=np.nan
            all_ts[l,c,0:tslen] = new['vals'].fillna(method='ffill').iloc[::-1]
            all_ts[l,c,tslen:] = np.nan
    return all_ts,all_count

def plot_ratings(all_ts,all_count,cat,level,n=4000,title=''):
    fig,ax = plt.subplots(1,2,figsize=(15,5))
    all_ts[all_ts==0]=np.nan
    samplesize = np.sum(np.isnan(all_ts)==False,axis=1)
    ts_mean = savgol_filter(np.nanmean(all_ts[:,:,0:n],axis=1).T,75,1,axis=0)
    ts_std = np.nanstd(all_ts[:,:,0:n],axis=1).T/np.sqrt(samplesize[:,:n].T)
    ax[0].plot(range(n),ts_mean)
    [ax[0].fill_between(range(n), ts_mean[:,i]-ts_std[:,i], ts_mean[:,i]+ts_std[:,i],alpha=.5) for i in range(2)]
    ax[0].set_title('Average Rating Across Time',fontsize=14)
    ax[0].set_xlabel('Days since restaurant opening')
    ax[0].set_ylabel('Ratings (exponentially weighted avg)')
    ax[0].legend(['visitors','locals'],prop={'size': 14},loc='upper center')

    all_count[all_count==0]=np.nan
    count_mean = savgol_filter(np.nanmean(all_count[:,:,0:n],axis=1).T,75,1,axis=0)
    count_std = np.nanstd(all_count[:,:,:n],axis=1).T/np.sqrt(samplesize[:,:n].T)
    ax[1].plot(range(n),count_mean)
    [ax[1].fill_between(range(n), count_mean[:,i]-count_std[:,i], count_mean[:,i]+count_std[:,i],alpha=.5) for i in range(2)]
    ax[1].set_title('Number of Reviews Across Time',fontsize=14)
    ax[1].set_xlabel('Days since restaurant opening')
    ax[1].set_ylabel('Number of Reviews (30 day rolling sum)')
    ax[1].legend(['visitors','locals'],prop={'size': 14},loc='upper center')
    if title is None:
    	fig.suptitle('{}={}'.format(cat,level),fontsize=18)
    else:
    	fig.suptitle(title,fontsize=18)

def bayes_avg(df,cond,col):
    v = df.groupby(cond)[col].size()
    m = df[cond].value_counts().mean()
    w = v/(v+m)
    S = w*df.groupby(cond)[col].mean() + (1-w)*df[col].mean()
    return S

def clean_words(text):
    text = re.sub(r'<.*?>','', text)
    words = re.sub('[^a-zA-Z]', ' ',text).lower().split()
    real_words = stopwords.words("english")
    extracted_words = [x for x in words if not x in real_words]
    paragraph = ' '.join(extracted_words)
    return(paragraph)

stemmer = PorterStemmer()

def stem_words(words_list, stemmer):
    return [stemmer.stem(word) for word in words_list if len(word)>5]

def tokenize(text):
    text = clean_words(text)
    tokens = word_tokenize(text)
    stems = stem_words(tokens, stemmer)
    return stems
