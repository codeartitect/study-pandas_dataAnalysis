# PMPDA - Part5

```
# 5-1 누락 데이터 확인

import seaborn as sns

df = sns.load_dataset('titanic')

# RaigeIndex : 인덱스 갯수, but 'deck' 열 갯수를 보면 203개 -> 누락 데이터가 688개라는 것이다.
print(df.info())

# 'deck'열 값 갯수 / dropna=True 유효 데이터 / dropna=False 누락 데이터
nan_deck = df['deck'].value_counts(dropna=False)
print(nan_deck)
print()

# isnull() 누락 데이터 -> True 반환
# df.head()를 누락인지 아닌지 확인
print(df.head().isnull())
print()

# notnull() 누락 데이터 -> False 반환
print(df.head().notnull())
print()

# df.head()에서 누락 데이터 갯수 구하기
print(df.head().isnull().sum(axis=0))

#### 출력
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
survived       891 non-null int64
pclass         891 non-null int64
sex            891 non-null object
age            714 non-null float64
sibsp          891 non-null int64
parch          891 non-null int64
fare           891 non-null float64
embarked       889 non-null object
class          891 non-null category
who            891 non-null object
adult_male     891 non-null bool
deck           203 non-null category
embark_town    889 non-null object
alive          891 non-null object
alone          891 non-null bool
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.6+ KB
None
NaN    688
C       59
B       47
D       33
E       32
A       15
F       13
G        4
Name: deck, dtype: int64

   survived  pclass    sex    age  ...   deck  embark_town  alive  alone
0     False   False  False  False  ...   True        False  False  False
1     False   False  False  False  ...  False        False  False  False
2     False   False  False  False  ...   True        False  False  False
3     False   False  False  False  ...  False        False  False  False
4     False   False  False  False  ...   True        False  False  False

[5 rows x 15 columns]

   survived  pclass   sex   age  ...   deck  embark_town  alive  alone
0      True    True  True  True  ...  False         True   True   True
1      True    True  True  True  ...   True         True   True   True
2      True    True  True  True  ...  False         True   True   True
3      True    True  True  True  ...   True         True   True   True
4      True    True  True  True  ...  False         True   True   True

[5 rows x 15 columns]

survived       0
pclass         0
sex            0
age            0 
sibsp          0
parch          0
fare           0
embarked       0
class          0
who            0
adult_male     0
deck           3
embark_town    0
alive          0
alone          0
dtype: int64
```

```
# 5-2 누락 데이터 제거

import seaborn as sns

df = sns.load_dataset('titanic')

missing_df = df.isnull()                # null 값인지 아닌지에 대한 데이터프레임

for col in missing_df.columns:
    missing_count = missing_df[col].value_counts()          # 데이터프레임의 컬럼 값 기준 반복문

    try:
        print(col, ': ', missing_count[True])               # NaN 값이 있으면 개수 출력

    except:
        print(col, ': ', 0)                                 # NaN 값이 없으면 0개 출력

print('----------')
df_thresh = df.dropna(axis=1, thresh=500)           # NaN 값이 500 이상인 열 모두 삭제
print(df_thresh.columns)                # 삭제된 열을 제외한 데이터프레임 출력
print('----------')

df_age = df.dropna(subset=['age'], how='any', axis=0)           # age 열에 나이 데이터가 없는 모든 행 삭제
print(len(df_age))

#### 출력
survived :  0
pclass :  0
sex :  0
age :  177
sibsp :  0
parch :  0
fare :  0
embarked :  2
class :  0
who :  0
adult_male :  0
deck :  688
embark_town :  2
alive :  0
alone :  0
----------
Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
       'embarked', 'class', 'who', 'adult_male', 'embark_town', 'alive',
       'alone'],
      dtype='object')
----------
714
```

```
# 5-3 평균으로 누락데이터 바꾸기

import seaborn as sns

df = sns.load_dataset('titanic')

print(df['age'].head(10))			# head 10개
print()

mean_age = df['age'].mean(axis=0)			# 평균 값 구하기
df['age'].fillna(mean_age, inplace=True) 	# 평균 값 채우기

print(df['age'].head(10))

#### 출력
0    22.0
1    38.0
2    26.0
3    35.0
4    35.0
5     NaN
6    54.0
7     2.0
8    27.0
9    14.0
Name: age, dtype: float64

0    22.000000
1    38.000000
2    26.000000
3    35.000000
4    35.000000
5    29.699118
6    54.000000
7     2.000000
8    27.000000
9    14.000000
Name: age, dtype: float64
```

```
# 5-4 가장 많이 나타나는 값으로 바꾸기

import seaborn as sns

df = sns.load_dataset('titanic')

print(df['embark_town'][825:830])           # 825~830 출력
print()

# value_counts : 열 안에 있는 값을 분류하여 그 값의 갯수를 구한다.
# idxmax : 최빈값 뽑아내기 - value_counts 와 반드시 같이 써야한다.
most_freq = df['embark_town'].value_counts(dropna=True).idxmax()
print(most_freq)
print()

#import numpy as np
# df['embark_town'].replace(np.nan, most_freq, inplace=True)
df['embark_town'].fillna(most_freq, inplace=True)           # 바로 위 두줄을 이용하여도 무관하다.

print(df['embark_town'][825:830])

#### 출력
825     Queenstown
826    Southampton
827      Cherbourg
828     Queenstown
829            NaN
Name: embark_town, dtype: object

Southampton

825     Queenstown
826    Southampton
827      Cherbourg
828     Queenstown
829    Southampton
Name: embark_town, dtype: object
```

```
# 5-5 이웃하고 있는 값으로 바꾸기

import seaborn as sns

df = sns.load_dataset('titanic')

print(df['embark_town'][825:830])
print()

# 바로 앞에 있는 828행의 값으로 변경
df['embark_town'].fillna(method='ffill', inplace=True)
print(df['embark_town'][825:830])

#### 출력
825     Queenstown
826    Southampton
827      Cherbourg
828     Queenstown
829            NaN
Name: embark_town, dtype: object

825     Queenstown
826    Southampton
827      Cherbourg
828     Queenstown
829     Queenstown
Name: embark_town, dtype: object
```

```
# 5-6 중복 데이터 확인

import pandas as pd

df = pd.DataFrame({'c1': ['a', 'a', 'b', 'a', 'a'],
                   'c2': [1, 1, 1, 2, 1],
                   'c3': [1, 1, 2, 2, 1]})
print(df)
print()

df_dup = df.duplicated()            # 모든 x행을 x-1행과 비교
print(df_dup)
print()

col_dup = df['c2'].duplicated()     # 'c2'의 x행을 x-1행과 비교
print(col_dup)      # x=0 일 때, -1이 마지막을 의미하지는 않는다.

#### 출력
  c1  c2  c3
0  a   1   1
1  a   1   1
2  b   1   2
3  a   2   2
4  a   1   1

0    False
1     True
2    False
3    False
4     True
dtype: bool

0    False
1     True
2     True
3    False
4     True
Name: c2, dtype: bool
```

```
# 5-7 중복 데이터 제거

import pandas as pd

df = pd.DataFrame({'c1': ['a', 'a', 'b', 'a', 'a'],
                   'c2': [1, 1, 1, 2, 2],
                   'c3': [1, 1, 2, 2, 2]})
print(df)
print()

df2 = df.drop_duplicates()
print(df2)
print()

df3 = df.drop_duplicates(subset=['c2', 'c3'])
print(df3)

#### 출력
  c1  c2  c3
0  a   1   1
1  a   1   1
2  b   1   2
3  a   2   2
4  a   2   2

  c1  c2  c3
0  a   1   1
2  b   1   2
3  a   2   2

  c1  c2  c3
0  a   1   1
2  b   1   2
3  a   2   2
```

```
# 5-8 단위 확산

import pandas as pd

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/auto-mpg.csv',
                 header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']
print(df.head(3))
print()

# 단위 변환 ->  mpg = mile per gallon / kpl = kilometer per liter
mpg_to_kpl = 1.60934 / 3.78541

df['kpl'] = df['mpg'] * mpg_to_kpl
print(df.head(3))
print()

df['kpl'] = df['kpl'].round(2)      # 소수점 둘째 자리에서 반올림
print(df.head(3))

#### 출력
    mpg  cylinders  displacement  ... model_year  origin                       name
0  18.0          8         307.0  ...         70       1  chevrolet chevelle malibu
1  15.0          8         350.0  ...         70       1          buick skylark 320
2  18.0          8         318.0  ...         70       1         plymouth satellite

[3 rows x 9 columns]

    mpg  cylinders  displacement  ... origin                       name       kpl
0  18.0          8         307.0  ...      1  chevrolet chevelle malibu  7.652571
1  15.0          8         350.0  ...      1          buick skylark 320  6.377143
2  18.0          8         318.0  ...      1         plymouth satellite  7.652571

[3 rows x 10 columns]

    mpg  cylinders  displacement  ... origin                       name   kpl
0  18.0          8         307.0  ...      1  chevrolet chevelle malibu  7.65
1  15.0          8         350.0  ...      1          buick skylark 320  6.38
2  18.0          8         318.0  ...      1         plymouth satellite  7.65

[3 rows x 10 columns]
```

```
# 5-9 자료형 변환

import pandas as pd
import numpy as np

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/auto-mpg.csv')

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']

print(df.dtypes)            # 각 열의 자료형 출력
print()

print(df['horsepower'].unique())            # df['horsepower']의 고유값 출력
print()

# '?'을 NaN으로 변환 -> NaN 제거 -> 실수형 변환
df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

print(df['horsepower'].dtypes)
print()

df['origin'].replace({1:'USA', 2:'EU', 3:'JAPAN'}, inplace=True)

print(df['origin'].unique())
print(df['origin'].dtypes)
print()

df['origin'] = df['origin'].astype('category')
print(df['origin'].dtypes)
print()

df['origin'] = df['origin'].astype('str')
print(df['origin'].dtypes)
print()

print(df['model_year'].sample(3))
print()
df['model_year'] = df['model_year'].astype('category')
print(df['model_year'].sample(3))

#### 출력
mpg             float64
cylinders         int64
displacement    float64
horsepower       object
weight          float64
acceleration    float64
model_year        int64
origin            int64
name             object
dtype: object

['165.0' '150.0' '140.0' '198.0' '220.0' '215.0' '225.0' '190.0' '170.0'
 '160.0' '95.00' '97.00' '85.00' '88.00' '46.00' '87.00' '90.00' '113.0'
 '200.0' '210.0' '193.0' '?' '100.0' '105.0' '175.0' '153.0' '180.0'
 '110.0' '72.00' '86.00' '70.00' '76.00' '65.00' '69.00' '60.00' '80.00'
 '54.00' '208.0' '155.0' '130.0' '112.0' '92.00' '145.0' '137.0' '158.0'
 '167.0' '94.00' '107.0' '230.0' '49.00' '75.00' '91.00' '122.0' '67.00'
 '83.00' '78.00' '52.00' '61.00' '93.00' '148.0' '129.0' '96.00' '71.00'
 '98.00' '115.0' '53.00' '81.00' '79.00' '120.0' '152.0' '102.0' '108.0'
 '68.00' '58.00' '149.0' '89.00' '63.00' '48.00' '66.00' '139.0' '103.0'
 '125.0' '133.0' '138.0' '135.0' '142.0' '77.00' '62.00' '132.0' '84.00'
 '64.00' '74.00' '116.0' '82.00']

float64

['USA' 'JAPAN' 'EU']
object

category

object

202    76
213    76
192    76
Name: model_year, dtype: int64

337    81
80     72
255    78
Name: model_year, dtype: category
Categories (13, int64): [70, 71, 72, 73, ..., 79, 80, 81, 82]
```

```
# 5-10 데이터 구간 분할

import pandas as pd
import numpy as np

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/auto-mpg.csv')

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

# 3개의 bin으로 나누는 경계 값의 리스트 구하기
count, bin_drivers = np.histogram(df['horsepower'], bins=3)
print(bin_drivers)
print()

# bin 이름 지정
bin_names = ['저출력', '보통출력', '고출력']

df['hp_bin'] = pd.cut(x=df['horsepower'],
                      bins=bin_drivers,
                      labels=bin_names,
                      include_lowest=True)

print(df[['horsepower', 'hp_bin']].head(15))

# 그래프 표현
import matplotlib.pyplot as plt
plt.rc('font', family='D2Coding')
plt.hist(df['hp_bin'])
plt.show()

#### 출력
[ 46.         107.33333333 168.66666667 230.        ]

    horsepower hp_bin
0        165.0   보통출력
1        150.0   보통출력
2        150.0   보통출력
3        140.0   보통출력
4        198.0    고출력
5        220.0    고출력
6        215.0    고출력
7        225.0    고출력
8        190.0    고출력
9        170.0    고출력
10       160.0   보통출력
11       150.0   보통출력
12       225.0    고출력
13        95.0    저출력
14        95.0    저출력
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200312192950056](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200312192950056.png) |

```
# 5-11 더미 변수

import numpy as np
import pandas as pd

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/auto-mpg.csv',
                 header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

# np.histogram(df['horsepower'], bins=3) 값이 두개임으로 변수 두개 필요
print(np.histogram(df['horsepower'], bins=3))
print()
count, bin_dividers = np.histogram(df['horsepower'], bins=3)

bin_name = ['저출력', '보통출력', '고출력']

df['hp_bin'] = pd.cut(x=df['horsepower'],
                      bins=bin_dividers,
                      labels=bin_name,
                      include_lowest=True)

# 더미 변수 : 크고 작음을 의미하는 것이 아니라 값이 존재하는지 안하는지에 대한 여부 값
horsepower_dummies = pd.get_dummies(df['hp_bin'])
print(horsepower_dummies.head(15))

#### 츨력
(array([257, 103,  32], dtype=int64), array([ 46.        , 107.33333333, 168.66666667, 230.        ]))

hp_bin  저출력  보통출력  고출력
0         0     1    0
1         0     1    0
2         0     1    0
3         0     1    0
4         0     1    0
5         0     0    1
6         0     0    1
7         0     0    1
8         0     0    1
9         0     0    1
10        0     0    1
11        0     1    0
12        0     1    0
13        0     0    1
14        1     0    0
```

```
# 원핫인코딩

import pandas as pd
import numpy as np

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/auto-mpg.csv', header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

count, bin_drivers = np.histogram(df['horsepower'], bins=3)

bin_name = ['저출력', '보통출력', '고출력']

df['hp_bin'] = pd.cut(x=df['horsepower'],
                      bins=bin_drivers,
                      labels=bin_name,
                      include_lowest=True)

from sklearn import preprocessing

# 객체 생성
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

# 1차원 벡터
onehot_labeled = label_encoder.fit_transform(df['hp_bin'].head(15))
print(onehot_labeled)
print(type(onehot_labeled))
print()

# 2차원 행렬
onehot_reshaped = onehot_labeled.reshape(len(onehot_labeled), 1)
print(onehot_reshaped)
print(type(onehot_reshaped))
print()

# 희소행렬
onehot_fitted = onehot_encoder.fit_transform(onehot_reshaped)
print(onehot_fitted)
print(type(onehot_fitted))


#### 출력
[1 1 1 1 1 0 0 0 0 0 0 1 1 0 2]
<class 'numpy.ndarray'>

[[1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [2]]
<class 'numpy.ndarray'>

  (0, 1)	1.0
  (1, 1)	1.0
  (2, 1)	1.0
  (3, 1)	1.0
  (4, 1)	1.0
  (5, 0)	1.0
  (6, 0)	1.0
  (7, 0)	1.0
  (8, 0)	1.0
  (9, 0)	1.0
  (10, 0)	1.0
  (11, 1)	1.0
  (12, 1)	1.0
  (13, 0)	1.0
  (14, 2)	1.0
<class 'scipy.sparse.csr.csr_matrix'>
```

원핫인코딩이란 머신러닝을 적용하기 위한 작업 중 하나이다. 컴퓨터는 문자를 인식하는 데에 있어서 어려움이 있어 이 문자를 수치화하는 것이다.

```
# 5-13 정규화

import pandas as pd
import numpy as np

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/auto-mpg.csv')

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

print(df.horsepower.describe())
print()

# abs() : 절대값
df.horsepower = df.horsepower / abs(df.horsepower.max())

print(df.horsepower.head())
print()
print(df.horsepower.describe)


#### 출력
count    391.000000
mean     104.404092
std       38.518732
min       46.000000
25%       75.000000
50%       93.000000
75%      125.000000
max      230.000000
Name: horsepower, dtype: float64

0    0.717391
1    0.652174
2    0.652174
3    0.608696
4    0.860870
Name: horsepower, dtype: float64

<bound method NDFrame.describe of 0      0.717391
1      0.652174
2      0.652174
3      0.608696
4      0.860870
         ...   
392    0.373913
393    0.226087
394    0.365217
395    0.343478
396    0.356522
Name: horsepower, Length: 391, dtype: float64>
```

```
# 5-14 정규화

import pandas as pd
import numpy as np

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/auto-mpg.csv')

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

print(df.horsepower.describe())
print()

# 최댓값과 최솟값 차이의 간격을 0과 1로 나타내기 위함
min_x = df.horsepower - df.horsepower.min()
min_max = df.horsepower.max() - df.horsepower.min()
df.horsepower = min_x / min_max

print(df.horsepower.head())
print()
print(df.horsepower.describe())


#### 출력
count    391.000000
mean     104.404092
std       38.518732
min       46.000000
25%       75.000000
50%       93.000000
75%      125.000000
max      230.000000
Name: horsepower, dtype: float64

0    0.646739
1    0.565217
2    0.565217
3    0.510870
4    0.826087
Name: horsepower, dtype: float64

count    391.000000
mean       0.317414
std        0.209341
min        0.000000
25%        0.157609
50%        0.255435
75%        0.429348
max        1.000000
Name: horsepower, dtype: float64
```

```
# 문자열을 Timestamp로 변환

import pandas as pd

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/stock-data.csv')

print(df.head())
print()
print(df.info())

df['new_Date'] = pd.to_datetime(df['Date'])

print(df.head())
print()
print(df.info())
print()
print(type(df['new_Date'][0]))

df.set_index('new_Date', inplace=True)
df.drop('Date', axis=1, inplace=True)

print(df.head())
print()
print(df.info())


#### 출력
         Date  Close  Start   High    Low  Volume
0  2018-07-02  10100  10850  10900  10000  137977
1  2018-06-29  10700  10550  10900   9990  170253
2  2018-06-28  10400  10900  10950  10150  155769
3  2018-06-27  10900  10800  11050  10500  133548
4  2018-06-26  10800  10900  11000  10700   63039

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20 entries, 0 to 19
Data columns (total 6 columns):
Date      20 non-null object
Close     20 non-null int64
Start     20 non-null int64
High      20 non-null int64
Low       20 non-null int64
Volume    20 non-null int64
dtypes: int64(5), object(1)
memory usage: 1.1+ KB
None
         Date  Close  Start   High    Low  Volume   new_Date
0  2018-07-02  10100  10850  10900  10000  137977 2018-07-02
1  2018-06-29  10700  10550  10900   9990  170253 2018-06-29
2  2018-06-28  10400  10900  10950  10150  155769 2018-06-28
3  2018-06-27  10900  10800  11050  10500  133548 2018-06-27
4  2018-06-26  10800  10900  11000  10700   63039 2018-06-26

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20 entries, 0 to 19
Data columns (total 7 columns):
Date        20 non-null object
Close       20 non-null int64
Start       20 non-null int64
High        20 non-null int64
Low         20 non-null int64
Volume      20 non-null int64
new_Date    20 non-null datetime64[ns]
dtypes: datetime64[ns](1), int64(5), object(1)
memory usage: 1.2+ KB
None

<class 'pandas._libs.tslibs.timestamps.Timestamp'>
            Close  Start   High    Low  Volume
new_Date                                      
2018-07-02  10100  10850  10900  10000  137977
2018-06-29  10700  10550  10900   9990  170253
2018-06-28  10400  10900  10950  10150  155769
2018-06-27  10900  10800  11050  10500  133548
2018-06-26  10800  10900  11000  10700   63039

<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 20 entries, 2018-07-02 to 2018-06-01
Data columns (total 5 columns):
Close     20 non-null int64
Start     20 non-null int64
High      20 non-null int64
Low       20 non-null int64
Volume    20 non-null int64
dtypes: int64(5)
memory usage: 960.0 bytes
None
```

```
# 5-16 Timestamp를 Period로 변환

import pandas as pd

dates = ['2019-01-01', '2020-03-01', '2021-06-01']

ts_dates = pd.to_datetime(dates)
print(ts_dates)
print()

# freq에 따라 기준이 다르다. 'D'의 경우 하루가 기준
pr_day = ts_dates.to_period(freq='D')
print(pr_day)
pr_month = ts_dates.to_period(freq='M')
print(pr_month)
pr_year = ts_dates.to_period(freq='Y')
print(pr_year)


#### 출력
DatetimeIndex(['2019-01-01', '2020-03-01', '2021-06-01'], dtype='datetime64[ns]', freq=None)

PeriodIndex(['2019-01-01', '2020-03-01', '2021-06-01'], dtype='period[D]', freq='D')
PeriodIndex(['2019-01', '2020-03', '2021-06'], dtype='period[M]', freq='M')
PeriodIndex(['2019', '2020', '2021'], dtype='period[A-DEC]', freq='A-DEC')
```

```
# 5-17 Timestamp 배열 만들기
# 도장처럼 딱!!! 그 순간을 의미

import pandas as pd

ts_ms = pd.date_range(start='2019-01-01',
                      end=None,
                      periods=6,        # 생성 개수
                      freq='MS',		# 월의 시작일 기준
                      tz='Asia/Seoul')    

print(ts_ms)
print()

ts_me = pd.date_range('2019-01-01',
                      periods=6,
                      freq='M',
                      tz='Asia/Seoul')
print(ts_me)
print()

ts_3m = pd.date_range('2019-01-01',
                      periods=6,
                      freq='3M',
                      tz='Asia/Seoul')
print(ts_3m)


#### 출력
DatetimeIndex(['2019-01-01 00:00:00+09:00', '2019-02-01 00:00:00+09:00',
               '2019-03-01 00:00:00+09:00', '2019-04-01 00:00:00+09:00',
               '2019-05-01 00:00:00+09:00', '2019-06-01 00:00:00+09:00'],
              dtype='datetime64[ns, Asia/Seoul]', freq='MS')

DatetimeIndex(['2019-01-31 00:00:00+09:00', '2019-02-28 00:00:00+09:00',
               '2019-03-31 00:00:00+09:00', '2019-04-30 00:00:00+09:00',
               '2019-05-31 00:00:00+09:00', '2019-06-30 00:00:00+09:00'],
              dtype='datetime64[ns, Asia/Seoul]', freq='M')

DatetimeIndex(['2019-01-31 00:00:00+09:00', '2019-04-30 00:00:00+09:00',
               '2019-07-31 00:00:00+09:00', '2019-10-31 00:00:00+09:00',
               '2020-01-31 00:00:00+09:00', '2020-04-30 00:00:00+09:00'],
              dtype='datetime64[ns, Asia/Seoul]', freq='3M')
```

```
# Period 배열 만들기
# 순간이 아니라 기간을 의미한다

import pandas as pd

pr_m = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='M')
print(pr_m)
print()

pr_h = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='H')
print(pr_h)
print()

pr_2h = pd.period_range(start='2019-01-01',
                        end=None,
                        periods=3,
                        freq='2H')
print(pr_2h)


#### 출력
PeriodIndex(['2019-01', '2019-02', '2019-03'], dtype='period[M]', freq='M')

PeriodIndex(['2019-01-01 00:00', '2019-01-01 01:00', '2019-01-01 02:00'], dtype='period[H]', freq='H')

PeriodIndex(['2019-01-01 00:00', '2019-01-01 02:00', '2019-01-01 04:00'], dtype='period[2H]', freq='2H')
```

```
# 5-19 날짜 데이터 분리

import pandas as pd

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/stock-data.csv')

df['new_Date'] = pd.to_datetime(df['Date'])
print(df.head())
print()

df['Year'] = df['new_Date'].dt.year
df['Monrth'] = df['new_Date'].dt.month
df['Day'] = df['new_Date'].dt.day
print(df.head())
print()

# dt와 period의 차이
# dt : 데이터만 가져온다
# period : 기간을 가진 데이터를 가져온다

df['Date_yr'] = df['new_Date'].dt.to_period(freq='A')
df['Date_m'] = df['new_Date'].dt.to_period(freq='M')
print(df.head())
print()

df.set_index('Date_m', inplace=True)
print(df.head())


#### 출력
         Date  Close  Start   High    Low  Volume   new_Date
0  2018-07-02  10100  10850  10900  10000  137977 2018-07-02
1  2018-06-29  10700  10550  10900   9990  170253 2018-06-29
2  2018-06-28  10400  10900  10950  10150  155769 2018-06-28
3  2018-06-27  10900  10800  11050  10500  133548 2018-06-27
4  2018-06-26  10800  10900  11000  10700   63039 2018-06-26

         Date  Close  Start   High    Low  Volume   new_Date  Year  Monrth  Day
0  2018-07-02  10100  10850  10900  10000  137977 2018-07-02  2018       7    2
1  2018-06-29  10700  10550  10900   9990  170253 2018-06-29  2018       6   29
2  2018-06-28  10400  10900  10950  10150  155769 2018-06-28  2018       6   28
3  2018-06-27  10900  10800  11050  10500  133548 2018-06-27  2018       6   27
4  2018-06-26  10800  10900  11000  10700   63039 2018-06-26  2018       6   26

         Date  Close  Start   High    Low  ...  Year Monrth  Day  Date_yr   Date_m
0  2018-07-02  10100  10850  10900  10000  ...  2018      7    2     2018  2018-07
1  2018-06-29  10700  10550  10900   9990  ...  2018      6   29     2018  2018-06
2  2018-06-28  10400  10900  10950  10150  ...  2018      6   28     2018  2018-06
3  2018-06-27  10900  10800  11050  10500  ...  2018      6   27     2018  2018-06
4  2018-06-26  10800  10900  11000  10700  ...  2018      6   26     2018  2018-06

[5 rows x 12 columns]

               Date  Close  Start   High  ...  Year  Monrth Day  Date_yr
Date_m                                    ...                           
2018-07  2018-07-02  10100  10850  10900  ...  2018       7   2     2018
2018-06  2018-06-29  10700  10550  10900  ...  2018       6  29     2018
2018-06  2018-06-28  10400  10900  10950  ...  2018       6  28     2018
2018-06  2018-06-27  10900  10800  11050  ...  2018       6  27     2018
2018-06  2018-06-26  10800  10900  11000  ...  2018       6  26     2018

[5 rows x 11 columns]
```

```
# 5-20 날짜 인덱스 활용

import pandas as pd

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part5/stock-data.csv')

df['new_Date'] = pd.to_datetime(df['Date'])
df.set_index('new_Date', inplace=True)

print(df.head())
print()
print(df.index)
print()

df_y = df['2018']
print(df_y.head())
print()

df_ym = df.loc['2018-07']
print(df_ym)
print()

df_ym_cols = df.loc['2018-07', 'Start':'High']
print(df_ym_cols)
print()

df_ymd = df['2018-07-02']
print(df_ymd)
print()

df_ymd_range = df['2018-06-25':'2018-06-20']
print(df_ymd_range)
print()

today = pd.to_datetime('2018-12-25')
df['time_delta'] = today - df.index
df.set_index('time_delta', inplace=True)
df_180 = df['180 days':'189 days']
print(df_180)


#### 출력
                  Date  Close  Start   High    Low  Volume
new_Date                                                  
2018-07-02  2018-07-02  10100  10850  10900  10000  137977
2018-06-29  2018-06-29  10700  10550  10900   9990  170253
2018-06-28  2018-06-28  10400  10900  10950  10150  155769
2018-06-27  2018-06-27  10900  10800  11050  10500  133548
2018-06-26  2018-06-26  10800  10900  11000  10700   63039

DatetimeIndex(['2018-07-02', '2018-06-29', '2018-06-28', '2018-06-27',
               '2018-06-26', '2018-06-25', '2018-06-22', '2018-06-21',
               '2018-06-20', '2018-06-19', '2018-06-18', '2018-06-15',
               '2018-06-14', '2018-06-12', '2018-06-11', '2018-06-08',
               '2018-06-07', '2018-06-05', '2018-06-04', '2018-06-01'],
              dtype='datetime64[ns]', name='new_Date', freq=None)

                  Date  Close  Start   High    Low  Volume
new_Date                                                  
2018-07-02  2018-07-02  10100  10850  10900  10000  137977
2018-06-29  2018-06-29  10700  10550  10900   9990  170253
2018-06-28  2018-06-28  10400  10900  10950  10150  155769
2018-06-27  2018-06-27  10900  10800  11050  10500  133548
2018-06-26  2018-06-26  10800  10900  11000  10700   63039

                  Date  Close  Start   High    Low  Volume
new_Date                                                  
2018-07-02  2018-07-02  10100  10850  10900  10000  137977

            Start   High
new_Date                
2018-07-02  10850  10900

                  Date  Close  Start   High    Low  Volume
new_Date                                                  
2018-07-02  2018-07-02  10100  10850  10900  10000  137977

                  Date  Close  Start   High    Low  Volume
new_Date                                                  
2018-06-25  2018-06-25  11150  11400  11450  11000   55519
2018-06-22  2018-06-22  11300  11250  11450  10750  134805
2018-06-21  2018-06-21  11200  11350  11750  11200  133002
2018-06-20  2018-06-20  11550  11200  11600  10900  308596

                  Date  Close  Start   High    Low  Volume
time_delta                                                
180 days    2018-06-28  10400  10900  10950  10150  155769
181 days    2018-06-27  10900  10800  11050  10500  133548
182 days    2018-06-26  10800  10900  11000  10700   63039
183 days    2018-06-25  11150  11400  11450  11000   55519
186 days    2018-06-22  11300  11250  11450  10750  134805
187 days    2018-06-21  11200  11350  11750  11200  133002
188 days    2018-06-20  11550  11200  11600  10900  308596
189 days    2018-06-19  11300  11850  11950  11300  180656
```

