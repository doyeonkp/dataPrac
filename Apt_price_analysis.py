#!/usr/bin/env python
# coding: utf-8
# %%
### Copyright by JoeEun Park in Inflearn
# # 전국 신규 민간 아파트 분양가격 동향

# ## 실습
# * 공공데이터 다운로드 후 주피터 노트북으로 로드하기
# * 판다스를 통해 데이터를 요약하고 분석하기
# * 데이터 전처리와 병합하기
# * 수치형 데이터와 범주형 데이터 다루기
# * 막대그래프(bar plot), 선그래프(line plot), 산포도(scatter plot), 상관관계(lm plot), 히트맵, 상자수염그림, swarm plot, 도수분포표, 히스토그램(distplot) 실습하기

# 
# ## 데이터셋
# * 다운로드 위치 : https://www.data.go.kr/dataset/3035522/fileData.do
# 
# ### 전국 평균 분양가격(2013년 9월부터 2015년 8월까지)
# * 전국 공동주택의 3.3제곱미터당 평균분양가격 데이터를 제공
# 
# ###  주택도시보증공사_전국 평균 분양가격(2019년 12월)
# * 전국 공동주택의 연도별, 월별, 전용면적별 제곱미터당 평균분양가격 데이터를 제공
# * 지역별 평균값은 단순 산술평균값이 아닌 가중평균값임

# %%


# 엑셀과 유사한 판다스 라이브러리를 불러옴
import pandas as pd


# %%


# 1.최근 분양가 파일 로드
# 2.로드한 데이터 df_last라는 변수에 담음
# 3. encode option을 주지 않으면 unicodeDecodeError 발생하니 옵션은 필수!
# 참고, euc-kr은 한글문자 조합 1만 1172자 중 2350자만 표현가능하다
# 그러나,UTF-8 cp949 인코딩을 사용하면, 1만 1172자 모두 커버 가능
df_last = pd.read_csv("data/주택도시보증공사_전국 신규 민간 아파트 분양가격 동향_20200331.csv", encoding="cp949")
df_last.shape
#(행, 열)


# %%


#head로 미리보기
# head 처럼 사용법을 잘모르는 함수가 있다면 shift + tab을 누르면 man 이나옴
df_last.head()


# %%


# tail로도 미리보기 가능
df_last.tail()


# ### 2015년 부터 최근까지의 데이터 로드
# 전국 평균 분양가격(2013년 9월부터 2015년 8월까지) 파일을 불러옵니다.
# df_first 라는 변수에 담고 shape로 행과 열의 갯수를 출력합니다.

# %%


df_first = pd.read_csv("data/전국 평균 분양가격(2013년 9월부터 2015년 8월까지).csv", encoding="cp949")
df_first.shape


# %%


df_first.head()


# %%


df_first.tail()


# ### 데이터 요약하기

# %%


df_last.info()
# 지역명, 규모구분, 연도, 월 등의 데이터 갯수와 분양가격의 데이터 겟수가 다르다.
# 이유는 결측치(NaN)가 있기 때문, 이 부분을 해결해주어야함


# ### 결측치 보기

# isnull 혹은 isna 를 통해 데이터가 비어있는지를 확인할 수 있습니다.
# 결측치는 True로 표시되는데, True == 1 이기 때문에 이 값을 다 더해주면 결측치의 수가 됩니다.

# %%


df_last.isnull()
# True == 1, False == 0
# 각 컬럼을 isnull을 통해 구한후, true의 갯수 즉 1이 몇번 더해졌는지 확인하면
# NaN(결측치)가 몇개인지 구할 수 있다.
df_last.isnull().sum()


# %%


# isna를 통해서도 가능
df_last.isna().sum()


# ### 데이터 타입 변경
# 분양가격이 object(문자) 타입으로 되어 있습니다. 
# 문자열 타입을 계산할 수 없기 때문에 수치 데이터로 변경해 줍니다. 
# 결측치가 섞여 있을 때 변환이 제대로 되지 않습니다. 
# 그래서 pd.to_numeric 을 통해 데이터의 타입을 변경합니다.

# %%


# 초기 분양가격의 데이타 타입이 object이기 때문에,
# 수치 계산이 불가하여 수치데이터로 변환이 필요 -> astype을 통해서
# 하지만 이것으로 데이터 타입변환시, 공백이 껴있으면 str으로 인식하기때문에 변환 에러가 발생
# 이와 같은 상황을 피하기 위해서, pandas의 to_numeric에 옵션을 주어 공백을 없애면서
# 수치데이터로 형식 변화가 가능해진다.
# df_last["분양가격(㎡)"].astype(int)

pd.to_numeric(df_last["분양가격(㎡)"], errors="coerce")
#위에서 변화한 값이 데이터 타입이 float64인 이유는 NaN의 데이터 타입이 float64이기 때문


# %%


#새로운 컬럼에 결측치를 제거하여 정제된 데이터를 넣어주자
df_last["분양가격"] = pd.to_numeric(df_last["분양가격(㎡)"], errors="coerce")
df_last["분양가격"]


# %%


df_last["분양가격"].mean()


# ### 평당분양가격 구하기
# 공공데이터포털에 올라와 있는 2013년부터의 데이터는 평당분양가격 기준으로 되어 있습니다.
# 분양가격을 평당기준으로 보기위해 3.3을 곱해서 "평당분양가격" 컬럼을 만들어 추가해 줍니다.

# %%


df_last["평당분양가격"] = df_last["분양가격"] * 3.3
df_last.head(1)


# %%


# 변경 전 칼럼인 분양가격(㎡) 컬럼을 요약
df_last["분양가격(㎡)"].describe()
#결과값:
# count 분양가격(㎡) 값은 데이터가 정제 되어있지 않기때문에
# nan 혹은 공백 또한 값으로 인정하여 카운트 했기때문에 정제된 데이터 셋보다 갯수가 많다.
# unique : 겹치지 않는 값의 갯수
# top : 가장 자주나오는 값
# freq : top(가장 많이 나오는 값)이 몇번 등장하였는지


# %%


# 수치데이터로 변경된 분양가격 컬럼을 요약
df_last["분양가격"].describe()


# ### 규모구분을 전용면적 컬럼으로 변경
# 규모구분 컬럼은 전용면적에 대한 내용이 있습니다. 전용면적이라는 문구가 공통적으로 들어가고 규모구분보다는 전용면적이 좀 더 직관적이기 때문에 전용면적이라는 컬럼을 새로 만들어주고 기존 규모구분의 값에서 전용면적, 초과, 이하 등의 문구를 빼고 간결하게 만들어 봅니다.
# 
# 이 때 str 의 replace 기능을 사용해서 예를들면 "전용면적 60㎡초과 85㎡이하"라면 "60㎡~85㎡" 로 변경해 줍니다.
# 
# * pandas 의 string-handling 기능을 좀 더 보고 싶다면 :
# https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling

# %%


df_last["규모구분"].unique()
# 결과 프린트를보면 행에 계속 반복하여, "전용면적"이라는 단어가 들어가있다.
# 이러한 경우 저 글자가 계속 데이터 용량에 잡이게 된다.
# 그리하면 처리 속도와 메모리를 계속 잡아 먹기때문에 불필요한것은 drop해주는것이 좋다


# %%


#df_last[index].replace("target", "")로 사용하면
#target 문자열이 100%일치 하지 않으면 replace가 되지 않음.
#따라서  pandas에서 제공하는 .str과 함께 쓰여진다
df_last["전용면적"] = df_last["규모구분"].str.replace("전용면적", "")
df_last["전용면적"] = df_last["전용면적"].str.replace("초과", "~")
df_last["전용면적"] = df_last["전용면적"].str.replace("이하", "")
df_last["전용면적"] = df_last["전용면적"].str.replace(" ", "").str.strip()
df_last["전용면적"]


# ### 필요없는 컬럼 제거하기
# drop을 통해 전처리 해준 컬럼을 제거합니다. pandas의 데이터프레임과 관련된 메소드에는 axis 옵션이 필요할 때가 있는데 행과 열중 어떤 기준으로 처리를 할 것인지를 의미합니다. 보통 기본적으로 0으로 되어 있고 행을 기준으로 처리함을 의미합니다. 메모리 사용량이 줄어들었는지 확인합니다.

# %%


df_last.head(1)


# %%


df_last.info()


# %%


# drop시 axis 주의!!!
# axis 0: 행, 1: 열
df_last = df_last.drop(["규모구분", "분양가격(㎡)"], axis=1)


# %%


# 컬럼 제거를 통해 메모리 사용량이 줄은것을 확인
df_last.info()


# ## groupby 로 데이터 집계하기
# groupby 를 통해 데이터를 그룹화해서 연산을 해봅니다.

# %%


#지역명으로 분양가 평균을 구하고 그래프로 표현해보자
# df.groupby(["인덱스로 사용할 컬럼명"])["계산할 컬럼 값"].연산()
# df.groupby? 를 run 해서 example을 확인하면서 사용 추천
df_last.groupby(["지역명"])["평당분양가격"].mean()
#df_last.groupby(["지역명"])["평당분양가격"].sum()
#df_last.groupby(["지역명"])["평당분양가격"].max()
df_last.groupby(["지역명"])["평당분양가격"].describe()


# %%


# 전용면적으로 분양가격을 구하자
df_last.groupby(["전용면적"])["평당분양가격"].mean()


# %%


#지역명, 전용면적으로 평당분양가격 평균을 구하자
df_last.groupby(["지역명","전용면적"])["평당분양가격"].mean()


# %%


# .unstack()을 쓰면, groupby([index1, index2])에서 마지막으로 들어가있는
# index2가 컬럼값으로 들어감
df_last.groupby(["전용면적", "지역명"])["평당분양가격"].mean().unstack().round()


# %%


# 연도, 지역명으로 평당 분양가격 평균 구하기
# unstack 뒤에 T를 하면 마지막 index로 넣어준것이 컬럼이 아니라 앞에 있는것이 컬럼값이 됌
df_last.groupby(["연도", "지역명"])["평당분양가격"]. mean().unstack().T


# ## pivot table 로 데이터 집계하기
# * groupby 로 했던 작업을 pivot_table로 똑같이 해봅니다.

# %%


# groupby와 pivot table의 가장 큰 차이점은, 결과물이 Series이냐 data frame이냐에 차이이다.
# Series로 결과물이 나오는것이 속도가 조금 더 빠르다.
# df_last.groupby(["지역명"])["평당분양가격"].mean() 와 같은 작업
# pivot_table의 기본 연산타입이 mean()임, 다른 연산으로 바꾸고 싶을경우 aggfunc="연산"의 옵션을 주면된다.
pd.pivot_table(df_last, index=["지역명"], values=["평당분양가격"])


# %%


# df_last.groupby(["전용면적"])["평당분양가격"].mean()와 같은 작업
pd.pivot_table(df_last, index="전용면적", values="평당분양가격")


# %%


# 지역명, 전용면적으로 평당분양가격의 평균을 구합니다.
#df_last.groupby(["전용면적", "지역명"])["평당분양가격"].mean().unstack().round()
# unstack은 index의 마지막 값이 columns로 바꾸어 주는것이였기 때문에 pivot에서는 columns를 지정해주면 된다.
# df_last.pivot_table(index=["전용면적", "지역명"], values="평당분양가격")
df_last.pivot_table(index="전용면적", columns="지역명", values="평당분양가격").round()


# %%


# 연도, 지역명으로 평당 분양가격 평균 구하기
# df_last.groupby(["연도", "지역명"])["평당분양가격"]. mean().unstack().T
# pd.pivot_table(df_last, index=["연도","지역명"], values="평당분양가격")
pd.pivot_table(df_last, index=["연도","지역명"], values="평당분양가격")


# ## 최근 데이터 시각화 하기
# ### 데이터시각화를 위한 폰트설정
# 한글폰트 사용을 위해 matplotlib의 pyplot을 plt라는 별칭으로 불러옵니다.

# %%


import matplotlib as mpl
print (mpl.get_cachedir ())


# %%


import matplotlib.pyplot as plt
plt.rc("font", family="AppleGothic")


# ### Pandas로 시각화 하기 - 선그래프와 막대그래프
# pandas의 plot을 활용하면 다양한 그래프를 그릴 수 있습니다.
# seaborn을 사용했을 때보다 pandas를 사용해서 시각화를 할 때의 장점은 미리 계산을 하고 그리기 때문에 속도가 좀 더 빠릅니다.

# %%


# 지역명으로 분양가격의 평균을 구하고 선그래프로 시각화 합니다.
# g = df_last.groupby(["지역명"])["평당분양가격"].mean()
g = df_last.groupby(["지역명"])["평당분양가격"].mean().sort_values(ascending=False)
g.plot()


# %%


# 지역명으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
# rot 글씨가 돌아가 있어 재대로 보고싶을떄 rot로 원상 복귀, figsize로 x축 글씨등 겹침의 문제 해결
g.plot.bar(rot=0, figsize=(10,3))


# %%


# 전용면적으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
df_last.groupby(["전용면적"])["평당분양가격"].mean().plot.bar()


# %%


# 연도별 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
df_last.groupby(["연도"])["평당분양가격"].mean().plot()


# ### box-and-whisker plot | diagram
# 
# * https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.boxplot.html
# 
# * [상자 수염 그림 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/%EC%83%81%EC%9E%90_%EC%88%98%EC%97%BC_%EA%B7%B8%EB%A6%BC)
# * 가공하지 않은 자료 그대로를 이용하여 그린 것이 아니라, 자료로부터 얻어낸 통계량인 5가지 요약 수치로 그린다.
# * 5가지 요약 수치란 기술통계학에서 자료의 정보를 알려주는 아래의 다섯 가지 수치를 의미한다.
# 
# 
# 1. 최솟값
# 1. 제 1사분위수
# 1. 제 2사분위수( ), 즉 중앙값
# 1. 제 3 사분위 수( )
# 1. 최댓값
# 
# * Box plot 이해하기 : 
#     * [박스 플롯에 대하여 :: -[|]- Box and Whisker](https://boxnwhis.kr/2019/02/19/boxplot.html)
#     * [Understanding Boxplots – Towards Data Science](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)

# %%


df_last.pivot_table(index="월", columns="연도", values="평당분양가격").plot.box()


# %%


p = df_last.pivot_table(index="월", columns=["연도", "전용면적"], values="평당분양가격")
p.plot.box(figsize=(15,3), rot=30)


# %%


p = df_last.pivot_table(index="연도", columns=["월"], values="평당분양가격")
p.plot(figsize=(15,3), rot=30)
plt.legend(loc=2, bbox_to_anchor=(-0.17, 1)) #망할 범주 박스가 내 그래프를 가릴때 쓰면 됨
# 이분 블로그 좋음: https://dailyheumsi.tistory.com/97


# ### Seaborn 으로 시각화 해보기

# %%


# 라이브러리 로드하기
import seaborn as sns

#구버전 주피터는 아래의 작업도 같이 필요
# %matplotlib inline


# %%


# barplot으로 지역별 평당분양가격을 그려봅니다.
# ci option = confidence interval 신뢰구간 (검은색 선으로 그래프에 표현됌), None 으로 지정하면 안그림
# seaborn의 장점은, 여러 통계 계산치를 일일이 직접 계산하여 데이터로 저장한 후 그래프로 표현하지 않고,
# 바로 그래프 생성 과정에서 연산을 넣어 그릴수 있다.
# default 연산 타입은 mean이다.
plt.figure(figsize=(10,3))
sns.barplot(data=df_last, x="지역명", y="평당분양가격", ci="sd")


# %%


# barplot으로 연도별 평당분양가격을 그려봅니다.
sns.barplot(data=df_last, x="연도", y="평당분양가격")


# %%


# lineplot으로 연도별 평당분양가격을 그려봅시다.
# hue 옵션을 통해 지역별로 다르게 표시해 봅시다.
plt.figure(figsize=(10,5))
sns.lineplot(data=df_last, x="연도", y="평당분양가격", hue="지역명")
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)


# %%


# relplot 으로 서브플롯 그리기
# relplot은 그릴수 있는 종류가 scatter 와 line밖에 없다.
# 이 옵션은 kind="그래프 종류"
# col의 옵션을 주어서 무엇을 기준으로 그래프를 분리해 그릴지 지정할수 있다.
# col_wrap옵션을 주지 않으면, 한줄에 모든 그래프를 붙여 그리기때문에 확인 하기 어렵다.
sns.relplot(data=df_last, x="연도", y="평당분양가격"
            ,kind="line", hue="지역명", col="지역명", col_wrap=4, ci=None)


# %%


# catplot 으로 서브플롯 그리기
sns.catplot(data=df_last, x="연도", y="평당분양가격", kind="bar", col="지역명", col_wrap=4)


# ### boxplot과 violinplot

# %%


# 연도별 평당분양가격을 boxplot으로 그려봅니다.
# 최솟값
# 제 1사분위수
# 제 2사분위수( ), 즉 중앙값
# 제 3 사분위 수( )
# 최댓값
sns.boxplot(data=df_last, x="연도", y="평당분양가격")


# %%


# hue옵션을 주어 전용면적별로 다르게 표시해봅시다.
plt.figure(figsize=(12,3))
sns.boxplot(data=df_last, x="연도", y="평당분양가격", hue="전용면적")


# %%


#연도별 평당분양가격을 violinplot으로 그려봅시다.
#boxplot을 보안한 violinplot
#box plot으로 볼때 어디에 데이터가 많이 분포되있는지 확인이 어렵다.
# violinplot은 boxplot처럼 1,2,3사분위도 알수 있으며, 데이터의 쏠림도 볼수 있다. 
sns.violinplot(data=df_last, x="연도", y="평당분양가격")


# ### lmplot과 swarmplot 

# %%


# 연도별 평당분양가격을 lmplot으로 그려봅니다. 
# hue 옵션으로 전용면적을 표현해 봅니다.
sns.lmplot(data=df_last, x="연도", y="평당분양가격", 
           hue="전용면적", col= "전용면적", col_wrap=3,
           x_jitter=.1)


# %%


# 연도별 평당분양가격을 swarmplot 으로 그려봅니다. 
# swarmplot은 범주형(카테고리) 데이터의 산점도를 표현하기에 적합합니다.
# swarmplot은 일일이 점을 찍기 때문에 오래걸린다. => 큰데이터로 swarmplot 그리는것은 적합하지 않을수 있다.
plt.figure(figsize=(15,3))
sns.swarmplot(data=df_last, x="연도", y="평당분양가격", hue="전용면적")


# ### 이상치 보기

# %%


# 평당분양가격의 최대값을 구해서 max_price 라는 변수에 담습니다.
df_last["평당분양가격"].describe()


# %%


max_price = df_last["평당분양가격"].max()
max_price


# %%


# 서울의 평당분양가격이 특히 높은 데이터가 있습니다. 해당 데이터를 가져옵니다.
df_last[df_last["평당분양가격"] == max_price]


# ### 수치데이터 히스토그램 그리기

# distplot은 결측치가 있으면 그래프를 그릴 때 오류가 납니다. 
# 따라서 결측치가 아닌 데이터만 따로 모아서 평당분양가격을 시각화하기 위한 데이터를 만듭니다.
# 데이터프레임의 .loc를 활용하여 결측치가 없는 데이터에서 평당분양가격만 가져옵니다.

# %%


# 수치데이터를 카테고리 형태로 바꾸는 작업을 binning 혹은 bucketing이라 한다.
df_last["평당분양가격"].hist(bins=20)
# h = df_last.hist(bins=10)


# %%


# 결측치가 없는 데이터에서 평당분양가격만 가져옵니다. 그리고 price라는 변수에 담습니다.
# .loc[행]
# .loc[행, 열]
# bins라는 옵션을 주면, 몇개의 통에다 데이터를 분리하여 담을지를 얘기한다.
# df_last[df_last["평당분양가격"].notnull()]
# loc을 사용하면, 평당분양가격이 null이 아닌데이터중, 평당분양가격만 보여줘의 뜻
# loc으로 원하는 정확한 데이커 행을 지정하지 않을경우 모든 columns에서 평당분양가격이 null이 아닌데이터 보여줌
price = df_last.loc[df_last["평당분양가격"].notnull(), "평당분양가격"]


# %%


# distplot으로 평당분양가격을 표현해 봅니다.
# sns.displot(df_last["평당분양가격"])
# 에러가 난다, 결측치가 제대로 처리가 않났기 때문
sns.distplot(price)
#결과 값을 확인해보면, 위의 히스토그램 그렸던 그래프와는 왼쪽(y-axis)의 값이 다른것을 확인할수 있다.
# 이유는 kde가 부드러운 곡선으로 그래프를 그릴수 있게 표시하는데, kde의 밀도가 1이 되는것을 찾아 그려준다.
# kde는 가우스분포를 추정하여 그린 그래프다.
# 가우스분포를 잘 모르겠어서 뭔말인지 모르겠다... 통계학 공부좀 해야겠다...ㅠㅠ


# %%


# hist=False하면 그래프 안에 나와 있는 바그래프가 사라지고 라인만 보인다.
# 러그 true하면 밑에 데이터의 빈도수도 눈으로 확인이 가능하다.
# sns.distplot(price, hist=False, rug=True)
sns.kdeplot(price, cumulative=True) #누적그래프 기록이 가능 (누적 판매량, 누적 거래량 등의 데이터에는 유용)


# * distplot을 산마루 형태의 ridge plot으로 그리기
# * https://seaborn.pydata.org/tutorial/axis_grids.html#conditional-small-multiples
# * https://seaborn.pydata.org/examples/kde_ridgeplot.html

# %%


# subplot 으로 표현해 봅니다.
g = sns.FacetGrid(df_last, row="지역명",
                 height=1.7, aspect=4,)
g.map(sns.distplot, "평당분양가격", hist=False, rug=True)


# %%


# pairplot
# test = df_last[df_last["연도"] < 2020]
df_last_notnull = df_last.loc[df_last["평당분양가격"].notnull(),
                          ["연도", "월", "평당분양가격", "지역명", "전용면적"]]
sns.pairplot(df_last_notnull, hue="지역명", diag_kws={'bw':0.4444})


# %%


df_last["전용면적"].value_counts()


# ## 2015년 8월 이전 데이터 보기

# %%


# col혹은 row가 ...으로 생략 되서 보일수도 있어 옵션을 주어 늘릴수 있다
pd.options.display.max_columns = 100


# %%


df_last.head()


# %%


df_first.head()


# %%


# 모든 컬럼이 출력되게 설정합니다.
df_first.info()


# %%


#결측치가 있는지 확인
df_first.isnull().sum()


# ### melt로 Tidy data 만들기
# pandas의 melt를 사용하면 데이터의 형태를 변경할 수 있습니다. 
# df_first 변수에 담긴 데이터프레임은 df_last에 담겨있는 데이터프레임의 모습과 다릅니다. 
# 같은 형태로 만들어주어야 데이터를 합칠 수 있습니다. 
# 데이터를 병합하기 위해 melt를 사용해 열에 있는 데이터를 행으로 녹여봅니다.
# 
# * https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-by-melt
# * [Tidy Data 란?](https://vita.had.co.nz/papers/tidy-data.pdf)

# %%


# head 로 미리보기 합니다.
df_first.head(1)


# %%


# pd.melt 를 사용하며, 녹인 데이터는 df_first_melt 변수에 담습니다. 
df_first_melt = df_first.melt(id_vars="지역", var_name="기간", value_name = "평당분양가격")


# %%


# df_first_melt 변수에 담겨진 컬럼의 이름을 
# ["지역명", "기간", "평당분양가격"] 으로 변경합니다.

df_first_melt.columns = ["지역명", "기간", "평당분양가격"]
df_first_melt.head(1)


# ### 연도와 월을 분리하기
# * pandas 의 string-handling 사용하기 : https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling

# %%


date = "2013년12월"
date


# %%


# split 을 통해 "년"을 기준으로 텍스트를 분리해 봅니다.
date.split("년")


# %%


# 리스트의 인덱싱을 사용해서 연도만 가져옵니다.
date.split("년")[0]


# %%


# 리스트의 인덱싱과 replace를 사용해서 월을 제거합니다.
date.split("년")[-1].replace("월","")


# %%


# parse_year라는 함수를 만듭니다.
# 연도만 반환하도록 하며, 반환하는 데이터는 int 타입이 되도록 합니다.
def parse_year(date):
    year = date.split("년")[0]
    return int(year)

y = parse_year(date)
y


# %%


# 제대로 분리가 되었는지 parse_year 함수를 확인합니다.
parse_year(date)


# %%


# parse_month 라는 함수를 만듭니다.
# 월만 반환하도록 하며, 반환하는 데이터는 int 타입이 되도록 합니다.
def parse_month(date):
    month = date.split("년")[-1].replace("월","")
    return int(month)


# %%


# 제대로 분리가 되었는지 parse_month 함수를 확인합니다.
parse_month(date)


# %%


# df_first_melt 변수에 담긴 데이터프레임에서 
# apply를 활용해 연도만 추출해서 새로운 컬럼에 담습니다.

df_first_melt["연도"] = df_first_melt['기간'].apply(parse_year)
df_first_melt["연도"]


# %%


# df_first_melt 변수에 담긴 데이터프레임에서 
# apply를 활용해 월만 추출해서 새로운 컬럼에 담습니다.

df_first_melt["월"] = df_first_melt['기간'].apply(parse_month)
df_first_melt.head(1)


# %%


# df_last와 병합을 하기 위해서는 컬럼의 이름이 같아야 합니다.
# sample을 활용해서 데이터를 미리보기 합니다.

df_last.sample()


# %%


# 컬럼명을 리스트로 만들때 버전에 따라 tolist() 로 동작하기도 합니다.
# to_list() 가 동작하지 않는다면 tolist() 로 해보세요.
df_last.columns.to_list()


# %%


cols = ['지역명', '연도', '월', '평당분양가격']
cols


# %%


# 최근 데이터가 담긴 df_last 에는 전용면적이 있습니다. 
# 이전 데이터에는 전용면적이 없기 때문에 "전체"만 사용하도록 합니다.
# loc를 사용해서 전체에 해당하는 면적만 copy로 복사해서 df_last_prepare 변수에 담습니다.
# .copy()를 하지 않게 되면 deepcopy가 아니기 때문에 원본 데이터 즉 df_last의 값이 변할수있다.

# df_last[df_last["전용면적"] == "전체"][cols]
df_last_prepare = df_last.loc[df_last["전용면적"] == "전체", cols].copy()
df_last_prepare.tail(1)


# %%


# df_first_melt에서 공통된 컬럼만 가져온 뒤
# copy로 복사해서 df_first_prepare 변수에 담습니다.

df_first_prepare = df_first_melt[cols].copy()
df_first_prepare


# ### concat 으로 데이터 합치기
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

# %%


# df_first_prepare 와 df_last_prepare 를 합쳐줍니다.

df = pd.concat([df_first_prepare, df_last_prepare])
df.shape


# %%


# 제대로 합쳐졌는지 미리보기를 합니다.
print(df.head())
print(df.tail())


# %%


# 연도별로 데이터가 몇개씩 있는지 value_counts를 통해 세어봅니다.
df["연도"].value_counts(sort=False)


# ### pivot_table 사용하기
# * https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-and-pivot-tables

# %%


# 연도를 인덱스로, 지역명을 컬럼으로 평당분양가격을 피봇테이블로 그려봅니다.
t = pd.pivot_table(df, index="연도", columns="지역명", values="평당분양가격").round()
t


# %%


# 위에서 그린 피봇테이블을 히트맵으로 표현해 봅니다.
plt.figure(figsize=(15,7))
sns.heatmap(t, cmap="Blues", annot=True, fmt=".0f")


# %%


# transpose 를 사용하면 행과 열을 바꿔줄 수 있습니다.

t.T


# %%


# 바뀐 행과 열을 히트맵으로 표현해 봅니다.

plt.figure(figsize=(15,7))
sns.heatmap(t.T, cmap="Blues", annot=True, fmt=".0f")


# %%


# Groupby로 그려봅니다. 인덱스에 ["연도", "지역명"] 을 넣고 그려봅니다.
g = df.groupby(["연도", "지역명"])["평당분양가격"].mean().unstack().round()
g


# %%


plt.figure(figsize=(15,7))
sns.heatmap(g.T, cmap="Greens", annot=True, fmt=".0f")


# ## 2013년부터 최근 데이터까지 시각화하기
# ### 연도별 평당분양가격 보기

# %%


# barplot 으로 연도별 평당분양가격 그리기
sns.barplot(data=df, x="연도", y="평당분양가격")


# %%


# pointplot 으로 연도별 평당분양가격 그리기
plt.figure(figsize=(12,4))
sns.pointplot(data=df, x="연도", y="평당분양가격", hue="지역명")
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)


# %%


# 서울만 barplot 으로 그리기
df_seoul = df[df["지역명"]=="서울"]
print(df_seoul.shape)

g = sns.barplot(data=df_seoul, x="연도", y="평당분양가격", color="b")
l = sns.pointplot(data=df_seoul, x="연도", y="평당분양가격")


# %%


# 연도별 평당분양가격 boxplot 그리기
sns.boxplot(data=df, x="연도", y="평당분양가격")


# %%


sns.boxenplot(data=df, x="연도", y="평당분양가격")


# %%


# 연도별 평당분양가격 violinplot 그리기
plt.figure(figsize=(10,4))
sns.violinplot(data=df,x="연도", y="평당분양가격", hue="지역명")


# %%


# 연도별 평당분양가격 swarmplot 그리기
# sns.lmplot(data=df, x="연도", y="평당분양가격")
plt.figure(figsize=(12,5))
sns.violinplot(data=df,x="연도", y="평당분양가격")
sns.swarmplot(data=df, x="연도", y="평당분양가격", hue="지역명")
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)


# ### 지역별 평당분양가격 보기

# %%


# barplot 으로 지역별 평당분양가격을 그려봅니다.
plt.figure(figsize=(12,4))
sns.barplot(data=df, x="지역명", y="평당분양가격")


# %%


# boxplot 으로 지역별 평당분양가격을 그려봅니다.
plt.figure(figsize=(12,4))
sns.boxplot(data=df, x="지역명", y="평당분양가격")


# %%


plt.figure(figsize=(12,4))
sns.boxenplot(data=df, x="지역명", y="평당분양가격")


# %%


# violinplot 으로 지역별 평당분양가격을 그려봅니다.
plt.figure(figsize=(25,5))
sns.violinplot(data=df, x="지역명", y="평당분양가격")


# %%


# swarmplot 으로 지역별 평당분양가격을 그려봅니다.
plt.figure(figsize=(25,5))
sns.swarmplot(data=df, x="지역명", y="평당분양가격", hue="연도")


# %%




