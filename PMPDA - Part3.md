# PMPDA - Part3

```
# 3-1 데이터 살펴보기

import pandas as pd

df = pd.read_csv('E:\Project\StudyProject\PMPDA material\part3/auto-mpg.csv', header=None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model_year','origin','name']

print(df.head())
print()
print(df.tail())
print()

print(df.shape)
print()

print(df.info())
print()

print(df.dtypes)
print()

print(df.mpg.dtypes)
print()

print(df.describe())
print()
print(df.describe(include='all'))

#### 출력
    mpg  cylinders  displacement  ... model_year  origin                       name
0  18.0          8         307.0  ...         70       1  chevrolet chevelle malibu
1  15.0          8         350.0  ...         70       1          buick skylark 320
2  18.0          8         318.0  ...         70       1         plymouth satellite
3  16.0          8         304.0  ...         70       1              amc rebel sst
4  17.0          8         302.0  ...         70       1                ford torino

[5 rows x 9 columns]

      mpg  cylinders  displacement  ... model_year  origin             name
393  27.0          4         140.0  ...         82       1  ford mustang gl
394  44.0          4          97.0  ...         82       2        vw pickup
395  32.0          4         135.0  ...         82       1    dodge rampage
396  28.0          4         120.0  ...         82       1      ford ranger
397  31.0          4         119.0  ...         82       1       chevy s-10

[5 rows x 9 columns]

(398, 9)

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 398 entries, 0 to 397
Data columns (total 9 columns):
mpg             398 non-null float64
cylinders       398 non-null int64
displacement    398 non-null float64
hoursepower     398 non-null object
weight          398 non-null float64
acceleration    398 non-null float64
model_year      398 non-null int64
origin          398 non-null int64
name            398 non-null object
dtypes: float64(4), int64(3), object(2)
memory usage: 28.1+ KB
None

mpg             float64
cylinders         int64
displacement    float64
hoursepower      object
weight          float64
acceleration    float64
model_year        int64
origin            int64
name             object
dtype: object

float64

              mpg   cylinders  ...  model_year      origin
count  398.000000  398.000000  ...  398.000000  398.000000
mean    23.514573    5.454774  ...   76.010050    1.572864
std      7.815984    1.701004  ...    3.697627    0.802055
min      9.000000    3.000000  ...   70.000000    1.000000
25%     17.500000    4.000000  ...   73.000000    1.000000
50%     23.000000    4.000000  ...   76.000000    1.000000
75%     29.000000    8.000000  ...   79.000000    2.000000
max     46.600000    8.000000  ...   82.000000    3.000000

[8 rows x 7 columns]

               mpg   cylinders  ...      origin        name
count   398.000000  398.000000  ...  398.000000         398
unique         NaN         NaN  ...         NaN         305
top            NaN         NaN  ...         NaN  ford pinto
freq           NaN         NaN  ...         NaN           6
mean     23.514573    5.454774  ...    1.572864         NaN
std       7.815984    1.701004  ...    0.802055         NaN
min       9.000000    3.000000  ...    1.000000         NaN
25%      17.500000    4.000000  ...    1.000000         NaN
50%      23.000000    4.000000  ...    1.000000         NaN
75%      29.000000    8.000000  ...    2.000000         NaN
max      46.600000    8.000000  ...    3.000000         NaN

[11 rows x 9 columns]
```

unique : 고유값 개수 / top : 최빈값 / freq : 빈도수

```
# 3-2 데이터 개수 확인

import pandas as pd

df = pd.read_csv('E:\Project\StudyProject\PMPDA material\part3/auto-mpg.csv')

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

print(df.count())
print()

print(type(df.count()))
print()

unique_values = df['origin'].value_counts()
print(unique_valuess)
print()

print(type(unique_values))

#### 출력
mpg             397
cylinders       397
displacement    397
horsepower      397
weight          397
acceleration    397
model year      397
origin          397
name            397
dtype: int64

<class 'pandas.core.series.Series'>

1    248
3     79
2     70
Name: origin, dtype: int64

<class 'pandas.core.series.Series'>
####
1 :미국을 나타내는 고유값
2: 유럽을 나타내는 고유값
3: 일본을 나타내는 고유값
```

```
# 3-3 통계 함수

import pandas as pd

df = pd.read_csv('E:\Project\StudyProject\PMPDA material\part3/auto-mpg.csv')

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','moder_year','origin','name']

print(df.mean())
print()
print(df['mpg'].mean())
print(df.mpg.mean())
print()
print(df[['mpg','weight']].mean())
print()

print(df.median())
print()
print(df['mpg'].median())
print()

print(df.max())
print()
print(df['mpg'].max())
print()

print(df.min())
print()
print(df['mpg'].min())
print()

print(df.std())
print()
print(df['mpg'].std())
print()

print(df.corr())
print()
print(df[['mpg','weight']].corr())

#### 출력

mpg               7.820926
cylinders         1.698329
displacement    104.244898
weight          847.485218
acceleration      2.755326
moder_year        3.689922
origin            0.802549
dtype: float64

7.820926403236991

                   mpg  cylinders  ...  moder_year    origin
mpg           1.000000  -0.775412  ...    0.578667  0.562894
cylinders    -0.775412   1.000000  ...   -0.344729 -0.561796
displacement -0.803972   0.950718  ...   -0.367470 -0.608749
weight       -0.831558   0.896623  ...   -0.305150 -0.580552
acceleration  0.419133  -0.503016  ...    0.284376  0.204102
moder_year    0.578667  -0.344729  ...    1.000000  0.178441
origin        0.562894  -0.561796  ...    0.178441  1.000000

[7 rows x 7 columns]

             mpg    weight
mpg     1.000000 -0.831558
weight -0.831558  1.000000
```

std : 표준편차 / corr : 상관계수

```
# 3-4 선 그래프 그리기

import pandas as pd
import matplotlib.pyplt as plt

df = pd.read_excel('E:\Project\StudyProject/PMPDA material/part3/남북한발전전력량.xlsx')

df_ns = df.iloc[[0,5],3:]
df_ns.index = ['South','North']
df_ns.columns = df_ns.columns.map(int)
print(df_ns.head())
print()

df_ns.plot()

tdf_ns = df_ns.T
print(tdf_ns.head())
print()
tdf_ns.plot()

plt.show()

#### 출력
       1991  1992  1993  1994  1995  1996  ...  2011  2012  2013  2014  2015  2016
South  1186  1310  1444  1650  1847  2055  ...  4969  5096  5171  5220  5281  5404
North   263   247   221   231   230   213  ...   211   215   221   216   190   239

[2 rows x 26 columns]

     South North
1991  1186   263
1992  1310   247
1993  1444   221
1994  1650   231
1995  1847   230
```

| 첫 그래프                                                    | 전치 후 그래프                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20200120145851181](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200120145851181.png) | ![image-20200120145858116](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200120145858116.png) |

.map(int) : 자료형 변경 - 연도가 문자열로 저장되있으므로 변경한 것임	

```
# 3-5,6 막대 그래프, 히스토그램

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('E:\Project\StudyProject/PMPDA material/part3/남북한발전전력량.xlsx')

df_ns = df.iloc[[0,5],3:]
df_ns.index = ['South','North']
df_ns.columns = df_ns.columns.map(int)

tdf_ns = df_ns.T
print(tdf_ns.head())
print()

tdf_ns.plot(kind='bar')
tdf_ns.plot(kind='hist')
plt.show()

#### 츨력
     South North
1991  1186   263
1992  1310   247
1993  1444   221
1994  1650   231
1995  1847   230
```

| 막대 그래프                                                  | 히스토그램                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20200120150720552](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200120150720552.png) | ![image-20200120150956216](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200120150956216.png) |

```
# 3-7,8 산점도, 박스 플롯

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('E:\Project\StudyProject\PMPDA material\part3/auto-mpg.csv')

df.columns = ['mpg','cylinders','displacement','horsepowe','weight','acceleration',
              'model_year','origin','name']

df.plot(x='weight',y='mpg',kind='scatter')
df[['mpg','cylinders']].plot(kind='box')
plt.show()
```

| 산점도                                                       | 박스 플롯                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20200120151921303](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200120151921303.png) | ![image-20200120151927283](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200120151927283.png) |