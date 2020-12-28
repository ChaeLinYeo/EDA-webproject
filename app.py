#!/usr/bin/env python
# coding: utf-8

# # 4. Exploratory Data Analysis
# 
# **탐색적 데이터 분석을 통해 데이터를 통달해봅시다.** with [Titanic Data](https://www.kaggle.com/c/titanic)
# 
# 0. 라이브러리 준비
# 1. 분석의 목적과 변수 확인
# 2. 데이터 전체적으로 살펴보기
# 3. 데이터의 개별 속성 파악하기

# # 탐색적 데이터 분석 - EDA
# ## EDA?
# 데이터를 분석하는 기술적 접근은 매우 많다. CNN, RNN, ...다양한 인공지능 기술들이 쏟아져나온다. 하지만 데이터가 가지는 본질적인 의미를 망각해서는 안된다. EDA는 데이터 그 자체에 적성과 특성을 요목조목 육안으로 확인하는 과정, 데이터 그 자체만으로부터 인사이트(시각화, 통계적 수치, numpy/pandas의 여러 컨테이너들)를 얻어내는 접근법이다!  
# 
# ## EDA의 Process
# 1. 분석의 목적(명확하게!)과 변수 확인(즉, column을 확인하는 것)
# 2. 데이터 전체적으로 살펴보기 (상관관계 분석, 결측치 즉 NA가 없는지)
# 3. 데이터의 개별 속성 파악하기(feature 등)
# 
# ## EDA with Example - Titanic
# https://www.kaggle.com/c/titanic  
# 머신러닝의 굉장히 유명한 데이터셋인 타이타닉 데이터셋이다. 데이터에서 얻을 수 있는 정보가 굉장히 많고, 적용해볼 수 있는 머신러닝 테크닉이 정말 많은 훌륭한 데이터셋이다. (컴퓨티 비전에서 자주 쓰이는 레나씨의 사진이 떠오른다..)  

# ## 0. 라이브러리 준비

# In[2]:


## 라이브러리 불러오기

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline # matplotlib 라이브러리를 인라인 환경에서 사용함을 반드시 명시')


# In[3]:


## 동일 경로에 "train.csv"가 있다면:
## 데이터 불러오기

titanic_df = pd.read_csv("./titanic/train.csv")


# ## 1. 분석의 목적과 변수 확인
# - 살아남은 사람들은 어떤 특징을 가지고 있었을까?  
# - Kaggle 사이트의 타이타닉 데이터셋에서 Data > Data Dictionary, Variable Notes를 확인한다.  

# In[5]:


## 상위 5개 데이터 확인하기
titanic_df.head(5)
# NaN은 결측치이다. 결측치는 중요한 단서이다. 이것을 메꿔야할 수도 있고, 이 결측치가 의미있는 것일수도 있다.


# In[6]:


## 각 Column의 데이터 타입 확인하기

titanic_df.dtypes
# object는 이름이나 성별이다.


# ## 2. 데이터 전체적으로 살펴보기

# In[8]:


## 데이터 전체 정보를 얻는함수 : .describe()

titanic_df.describe() # 수치형 데이터에 대한 요약만을 제공
# 따라서 아까에 비해 column이 줄어들었다!


# In[9]:


## 상관계수 확인!

titanic_df.corr() # 상관계수 행렬 출력
# Pclass와 Survived의 상관계수도 눈여겨볼 만 하다.
# Pclass와 Fare의 경우 비쌀수록 높은 클래스의 좌석을 이용했을 것이므로 음의 큰 상관관계가 나온다.

# Correlation is NOT Causation

# 상관성 : A up, B up, ... (A가 증가하면 B도 증가하는 경향성 등을 나타내는 수치)
# 인과성 : A -> B (A로부터 B가 발생한다는 종속관계를 의미)
# 이 두가지를 꼭 구분해서 사용해야 한다.
# 상관계수가 유의미하게 나왔다고 해서 이 둘 사이에 인과성이 꼭 존재하는 것은 아니다.


# In[11]:


## 결측치 확인

titanic_df.isnull().sum()
# Age, Cabin, Embarked에 결측치 발견!


# ## 3. 데이터의 개별 속성 파악하기

# ### 1. Survived Column

# In[17]:


## 생존자, 사망자 명수는?

titanic_df['Survived'].sum() # 생존자 명수


# In[16]:


titanic_df['Survived'].value_counts() # 사망자, 생존자 명수


# In[19]:


## 생존자수와 사망자수를 Barplot으로 그려보기
## sns.countplot()사용. seaborn이 깔끔하다~

sns.countplot(x='Survived', data=titanic_df) # 0은 사망자, 1은 생존자
plt.show()


# ### 2. Pclass

# In[21]:


# Pclass에 따른 인원 파악
titanic_df[['Pclass', 'Survived']].groupby(['Pclass']).count()


# In[23]:


# Pclass에 따른 생존자 인원은 어떻게 알 수 있을까?
titanic_df[['Pclass', 'Survived']].groupby(['Pclass']).sum()
# survived가 1인 개수를 이렇게 셀 수 있다!


# In[24]:


# 생존 비율?
# sum/count이다.
titanic_df[['Pclass', 'Survived']].groupby(['Pclass']).mean()
# Pclass가 높을수록 생존률이 높은 상관관계가 있음을 알 수 있다.


# In[25]:


# 히트맵 활용
sns.heatmap(titanic_df[['Pclass','Survived']].groupby(['Pclass']).mean())
plt.plot()


# ### Sex

# In[27]:


titanic_df[['Sex', 'Survived']]


# In[30]:


# groupby의 기준을 두개를 적용시키기.
titanic_df.groupby(['Survived', 'Sex'])['Survived'].count()


# In[32]:


# sns.catplot
# col : survived에 대한 케이스 분류
# x : 가로축 plot에 대한 기준
# kind : countplot을 이용함
sns.catplot(x='Sex', col='Survived', kind='count', data=titanic_df)
plt.show()
# 인사이트 : 남성이 더 많이 죽음, 여성이 더 많이 살아남음


# ### 4. Age
# #### Remind : 결측치 존재!

# In[34]:


titanic_df.describe()['Age']


# In[40]:


## Survived 1, 0과 Age의 경향성

fig, ax = plt.subplots(1, 1, figsize=(10, 5)) 
# 1, 1는 가로엔 몇개, 세로엔 몇개의 그래프를 그릴 것인지.
# 한 axis 위에 두개의 그래프를 그릴 것이다.
sns.kdeplot(x=titanic_df[titanic_df.Survived == 1]['Age'], ax=ax)
sns.kdeplot(x=titanic_df[titanic_df.Survived == 0]['Age'], ax=ax)

plt.legend(['Survived', 'Dead']) # 범위
 
plt.show()

# 인사이트 : 어린아이들 경우 생존 비율이 높음. 20~30대 청년들은 사망 비율이 높음. 고령층의 경우 사망 비율이 높다.


# ### Appendix 1. Sex + Pclass vs Survived

# In[41]:


sns.catplot(x='Pclass', y='Survived', kind='point', data=titanic_df)
plt.show()
# 포인트 그래프의 경우 점이 추정치를 의미한다.
# 막대기는 신뢰구간이다.
# Pclass가 1일수록 생존률이 높음을 알 수 있다.


# In[42]:


sns.catplot(x='Pclass', y='Survived', hue='Sex', kind='point', data=titanic_df)
plt.show()
# hue를 성별로 줌으로써 두가지 그래프를 그릴 수 있다.
# 여성이면서 1등석이면 생존률이 거의 1, 매우 높다. 반면 남성은 낮다.
# 여러 컬럼에 따라 이렇게 분석해보는 것도 중요하다.


# ## Appendix 2. Age + Pclass

# In[47]:


## Age graph with Pclass

titanic_df['Age'][titanic_df.Pclass == 1].plot(kind='kde')
titanic_df['Age'][titanic_df.Pclass == 2].plot(kind='kde')
titanic_df['Age'][titanic_df.Pclass == 3].plot(kind='kde')

plt.legend(['1st class', '2nd class', '3rd class']) # 범주
# Pclass별로 Age 그래프를 그릴 수 있다.

# 인사이트 : 높은 클래스일수록 나이대가 더 높아진다.


# ## Mission : It's Your Turn!
# 
# ### 1. 본문에서 언급된 Feature를 제외하고 유의미한 Feature를 1개 이상 찾아봅시다.
# 
# - Hint : Fare? Sibsp? Parch?
# 
# ### 2. [Kaggle](https://www.kaggle.com/datasets)에서 Dataset을 찾고, 이 Dataset에서 유의미한 Feature를 3개 이상 찾고 이를 시각화해봅시다.
# 
# 함께 보면 좋은 라이브러리 document
# - [numpy]()
# - [pandas]()
# - [seaborn]()
# - [matplotlib]()

# 무대뽀로 하기 힘들다면? 다음 Hint와 함께 시도해봅시다:
# 1. 데이터를 톺아봅시다.
# - 각 데이터는 어떤 자료형을 가지고 있나요?
# - 데이터에 결측치는 없나요? -> 있다면 이를 어떻게 메꿔줄까요?
# - 데이터의 자료형을 바꿔줄 필요가 있나요? -> 범주형의 One-hot encoding
# 2. 데이터에 대한 가설을 세워봅시다.
# - 가설은 개인의 경험에 의해서 도출되어도 상관이 없습니다.
# - 가설은 명확할 수록 좋습니다. ex) Titanic Data에서 Survival 여부와 성별에는 상관관계가 있다!
# 3. 가설을 검증하기 위한 증거를 찾아봅시다.
# - 이 증거는 한 눈에 보이지 않을 수 있습니다. 우리가 다룬 여러 Technique를 써줘야 합니다.
# - `groupby()`를 통해서 그룹화된 정보에 통계량을 도입하면 어떨까요?
# - 시각화를 통해 일목요연하게 보여주면 더욱 좋겠죠?

# In[269]:


# 라이브러리 선언
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[270]:


# 데이터 불러오기
data = pd.read_csv('./bestsellers with categories.csv')
data.head()


# In[271]:


# Non Fiction과 Fiction장르의 가격차이

plt.hist(data[data['Genre']=='Non Fiction']['Price'])
plt.hist(data[data['Genre']=='Fiction']['Price'])
plt.legend(['Non Ficton', 'Fiction'])

plt.show()
# 전반적으로 Non Fiction인 책이 Fiction인 책보다 가격대가 높음을 알 수 있다.
# Non Fiction인 책의 경우 전문서적 등이 많이 포함되어 Fiction인 책보다 가격대가 높은 게 않을까?


# In[272]:


# Non Fiction과 Fiction장르의 리뷰차이
sns.heatmap(data[['Genre','Reviews']].groupby(['Genre']).mean(), annot=True)
plt.show()
# Fiction이 Non Fiction보다 리뷰가 많은 것을 알 수 있다.


# In[273]:


sns.heatmap(data.corr(), annot=True)
plt.show()


# # 음.. 딱히 건져낼만한 유의미한 Feature을 더 찾기 힘들어서 다른 데이터셋으로 해보았다. 강한 상관관계가 보이지 않는다.
# ### 상관계수에 대하여
# - -1에 가까운 값이 얻어지면 : 누가 봐도 매우 강력한 음(-)의 상관. 오히려 너무 확고하기 때문에 사회과학 데이터일 경우 데이터를 조작한 게 아닌가 의심할 정도이다. 물론 이건 사회과학 얘기고 순수학문에 가까운 분야일수록 요구되는 상관관계는 높은 편.
# - -0.5 정도의 값이 얻어지면 : 강력한 음(-)의 상관. 연구자는 변인 x 가 증가하면 변인 y 가 감소한다고 자신 있게 말할 수 있다.
# - -0.2 정도의 값이 얻어지면 : 음(-)의 상관이긴 한데 너무 약해서 모호하다. 상관관계가 없다고는 할 수 없지만 좀 더 의심해 봐야 한다.
# - 0 정도의 값이 얻어지면 : 대부분의 경우, 상관관계가 있을거라고 간주되지 않는다. 다른 후속 연구들을 통해 뒤집어질지는 모르지만 일단은 회의적이다. 하지만 무조건적으로 그런건 아니라 2차 방정식 그래프와 비슷한 모양이 될 경우 상관관계는 있으나 상관계수는 0에 가깝게 나온다.
# - 0.2 정도의 값이 얻어지면 : 너무 약해서 의심스러운 양(+)의 상관. 이것만으로는 상관관계에 대해 아주 장담할 수는 없다. 하지만 사회과학에선 매우 큰 상관관계가 있는 것으로 간주한다.
# - 0.5 정도의 값이 얻어지면 : 강력한 양(+)의 상관. 변인 x 가 증가하면 변인 y 가 증가한다는 주장은 이제 통계적으로 지지받고 있다.
# - 1에 가까운 값이 얻어지면 : 이상할 정도로 강력한 양(+)의 상관. 위와 마찬가지로, 이렇게까지 확고한 상관관계는 오히려 쉽게 찾아보기 어렵다.
# 
# # Word Happiness Report 데이터셋
# ### https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
# - pandas API레퍼런스
# ### http://seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot
# - seaborn API레퍼런스

# In[274]:


data15 = pd.read_csv('./word_happiness_report/2015.csv')
data16 = pd.read_csv('./word_happiness_report/2016.csv')
data17 = pd.read_csv('./word_happiness_report/2017.csv')
data18 = pd.read_csv('./word_happiness_report/2018.csv')
data19 = pd.read_csv('./word_happiness_report/2019.csv')
# data = pd.concat([data15,data16,data17,data18,data19])
# data
# sns.heatmap(data.corr(), annot=True)
# 위와 같이 5개의 데이터를 한번에 합쳤더니 column도 다 다르고, 결측치도 많아서 엄청나게 더러워졌다 ㅠ
# 일단 각 column을 살펴보면서 정리해줘야겠다! 버릴 건 drop하고, 의미가 같지만 이름이 다른 column끼리는 통합해주고.


# In[275]:


data15.head(5)
# Standard Error는 2015년에만 존재한다.
# Dystopia Residual도 2015년과 2016년에만 존재한다. 2017년의 Dystopia.Residual와 같다.
# Region은 2015, 2016년에만 있다.
# Country는 다른 연도의 Country or region과 같은 내용을 담고 있다.
# Overall rank == Happiness Rank


# In[276]:


data16.head(5)
# Dystopia Residual도 2015년과 2016년에만 존재한다. 2017년의 Dystopia.Residual와 같다.
# Region은 2015, 2016년에만 있다.
# Country는 다른 연도의 Country or region과 같은 내용을 담고 있다.
# Lower Confidence Interval은 2016년에만 있다.
# Upper Confidence Interval도 2016년에만 있다.
# Overall rank == Happiness Rank


# In[277]:


data17.head(5)
# Country는 다른 연도의 Country or region과 같은 내용을 담고 있다.
# Whisker.high는 2017년에만 있다.
# Whisker.low도 2017년에만 있다.
# Dystopia.Residual는 2017년에만 있다. 2015,1016년의 Dystopia Residual와 같다.
# Overall rank == Happiness Rank


# In[278]:


data18.head(5)
# Country or region은 다른 연도의 Country와 같은 내용을 담고 있다.
# Overall rank == Happiness Rank


# In[279]:


data19.head(5)
# Country or region은 다른 연도의 Country와 같은 내용을 담고 있다.
# Overall rank == Happiness Rank


# ### 이제 모든 연도에 있지 않은 column들은 삭제해줄건데, del과 drop중 뭐를 쓰는게 좋을지 몰라서 찾아봤다.
# del은 "remove an item from a list"라며 list를 기준으로 remove가 일어납니다. drop은 pandas.DataFrame.drop method로서 pandas에 특화되어 있기 때문에 drop을 사용하는걸 추천드립니다.  
# 라고 나오길래 drop을 쓰기로 했다..!  
# 사용된 파라미터는 다음과 같다(판다스 API레퍼런스에 의하면..)  
# - columnssingle label or list-like
#     - Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).
# - nplacebool, default False
#     - If False, return a copy. Otherwise, do operation inplace and return None.
# - errors{‘ignore’, ‘raise’}, default ‘raise’
#     - If ‘ignore’, suppress error and only existing labels are dropped.
# - axis{0 or ‘index’, 1 or ‘columns’}, default 0
#     - Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).

# In[280]:


data15.drop(columns="Standard Error",inplace=True,errors="ignore", axis=1)
data15.drop(columns="Dystopia Residual",inplace=True,errors="ignore", axis=1)
data15.drop(columns="Region",inplace=True,errors="ignore", axis=1)

data16.drop(columns="Lower Confidence Interval",inplace=True,errors="ignore", axis=1)
data16.drop(columns="Upper Confidence Interval",inplace=True,errors="ignore", axis=1)
data16.drop(columns="Dystopia Residual",inplace=True,errors="ignore", axis=1)
data16.drop(columns="Region",inplace=True,errors="ignore", axis=1)

data17.drop(columns="Whisker.high",inplace=True,errors="ignore", axis=1)
data17.drop(columns="Whisker.low",inplace=True,errors="ignore", axis=1)
data17.drop(columns="Dystopia.Residual",inplace=True,errors="ignore", axis=1)


# ### 이제 모든 연도에 공통적으로 있는 column들만 남았다.  근데 같은 내용을 담고 있는 column들이라도 이름이 다른 경우가 있어 통일해줘야한다(노가다)

# In[281]:


data15=data15.rename(columns={"Country" : "Country or region",
                            "Economy (GDP per Capita)":"GDP per capita",
                            "Health (Life Expectancy)":"Healthy life expectancy",
                            "Trust (Government Corruption)":"Perceptions of corruption"})


# In[282]:


data16=data16.rename(columns={"Country" : "Country or region",
                            "Economy (GDP per Capita)":"GDP per capita",
                            "Health (Life Expectancy)":"Healthy life expectancy",
                            "Trust (Government Corruption)":"Perceptions of corruption"})


# In[283]:


data17=data17.rename(columns={"Country" : "Country or region",
                            "Happiness.Score":"Happiness Score",
                            "Economy..GDP.per.Capita.":"GDP per capita",
                            "Health..Life.Expectancy.":"Healthy life expectancy",
                            "Trust..Government.Corruption.":"Perceptions of corruption",
                             "Happiness.Rank" : "Happiness Rank"})


# In[284]:


data18=data18.rename(columns={"Freedom to make life choices" : "Freedom",
                             "Score" : "Happiness Score",
                             "Social support" : "Family",
                             "Overall rank" : "Happiness Rank"})


# In[285]:


data19=data19.rename(columns={"Freedom to make life choices" : "Freedom",
                             "Score" : "Happiness Score",
                             "Social support" : "Family",
                             "Overall rank" : "Happiness Rank"})


# In[286]:


alldata = pd.concat([data15,data16,data17,data18,data19])
alldata


# In[287]:


sns.heatmap(alldata.corr(), annot=True)


# ### 이제 깔끔하게 2015~20019년도를 합쳐서 column간의 상관관계를 파악할 수 있다!!
# ### 강한 상관관계를 갖는 column들 중, -0.5 또는 0.5에 가까운 값을 갖는 것들
# - Happiness Score & Freedom : 0.55
# - GDP per capita & Family : 0.59
# - Family & Freedom : 0.42
# - Healthy life expectanacy & Family : 0.57
# - Freedom & Perception of corruption : 0.46
# - Freedom & Happiness Rank : -0.54

# ### 1. Happiness Score & Freedom : 0.55

# In[293]:


sns.scatterplot(x='Happiness Score', y='Freedom', hue='Freedom', data=alldata)
plt.xlabel('Happiness Score')
plt.ylabel('Freedom')
plt.show()

# 행복지수가 높을수록 자유롭다고 느끼는 사람이 많음을 알 수 있다.


# ### 2. GDP per capita & Family : 0.59

# In[289]:


plt.scatter(x=alldata['GDP per capita'], y=alldata['Family'])
plt.xlabel('GDP per capita')
plt.ylabel('Family')
plt.show()

# 1인당 GDP 지수가 높을수록 가족이 많은 경우가 많음을 알 수 있다.


# ### 3. Family & Freedom : 0.42

# In[324]:


plt.xlabel('Family')
plt.ylabel('Freedom')
plt.hist(x=alldata['Family'], weights=alldata['Freedom'], bins=np.arange(0, 2, 0.1))
plt.xticks(np.arange(0, 2, 0.1)) # Extra : xticks를 올바르게 처리해봅시다.
plt.show()

# 가족이 많을수록 자유를 느끼는 사람들이 많음을 알 수 있다.


# ### 4. Healthy life expectancy & Family : 0.57

# In[307]:


sns.displot(data=alldata, x='Family', y='Healthy life expectancy')
plt.show()

# 가족이 많을수록, 건강한 삶에 대한 기대치가 높음을 알 수 있다.


# ### 5. Freedom & Perceptions of corruption : 0.46

# In[342]:


sns.jointplot(alldata['Freedom'],alldata['Perceptions of corruption'],kind="hex",size=7,ratio=3)
plt.show()

# 자유도가 높다고 해서 반드시 국가가 부패했다는 인식이 높은 것은 아니다.
# 하지만 비교적 자유도가 높을 때, 자유도가 낮을 때보다 국가가 부패했다는 비판적인 인식을 갖고 있는 사람들이 많음을 알 수 있다.
# 그리고 자유도가 높을수록 국가의 부패인식 정도가 0부터 0.5까지 넓게 분포하는 것을 통해, 
# 자유도가 높을수록 국가의 부패인식에 대해 다양한 생각을 갖고 있는 사람이 많음을 알 수 있다.


# ### 6. Freedom & Happiness Rank : -0.54

# In[345]:


sns.regplot(alldata['Freedom'],alldata['Happiness Rank'], color="g")
plt.show()
# 자유도가 높을수록 비교적 행복 순위가 떨어지는 경향을 보임을 알 수 있다.


# In[ ]:




