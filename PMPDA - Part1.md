# PMPDA - Part1

```
# 1-1 딕셔너리 -> 시리즈 변환

import pandas as pd

dict_data = {'a':1, 'b':2, 'c':3}
sr = pd.Series(dict_data)
print(type(dict_data))
print(sr)

#### 출력
<class 'pandas.core.series.Series'>
a    1
b    2
c    3
dtype: int64
```

```
# 1-2 시리즈 인덱스

import pandas as pd

list_data = ['2019-01-02',3.14,'ABC',100,True]
sr = pd.Series(list_data)
print(type(sr))
print(sr)

idx = sr.index
val = sr.values
print(idx)
print(val)

#### 출력
<class 'pandas.core.series.Series'>
0    2019-01-02
1          3.14
2           ABC
3           100
4          True
dtype: object
RangeIndex(start=0, stop=5, step=1)
['2019-01-02' 3.14 'ABC' 100 True]
```

```
# 1-3 시리즈 원소 선택

import pandas as pd

tup_data = ('영인', '2010-05-01', '여', True)
sr = pd.Series(tup_data, index=['이름', '생년월인', '성별', '학생여부'])
print(sr)
print(sr[0])
print(sr['이름'])

#### 출력
이름              영인
생년월인    2010-05-01
성별               여
학생여부          True
dtype: object
영인
영인
```

```
# 1-4 딕셔너리 -> 데이터프레임 변환

import pandas as pd

dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df = pd.DataFrame(dict_data)

print(type(df))
print(df)

#### 출력
<class 'pandas.core.frame.DataFrame'>
   c0  c1  c2  c3  c4
0   1   4   7  10  13
1   2   5   8  11  14
2   3   6   9  12  15
```

```
# 1-5 행 인덱스/열 이름 설정

import pandas as pd

df = pd.DataFrame([[15,'남','덕명중'], [17,'여','수리중']],
                  index=['준서','예은'],
                  columns=['나이','성별','학교'])

print(df)
print()
print(df.index)
print(df.columns)

df.index = ['학생1', '학생2']				# 인덱스, 컬럼값 변경
df.columns = ['연령', '남녀', '소속']				# 다음 코드창과 비교

print()
print(df)

#### 출력
    나이 성별   학교
준서  15  남  덕명중
예은  17  여  수리중

Index(['준서', '예은'], dtype='object')
Index(['나이', '성별', '학교'], dtype='object')

     연령 남녀   소속
학생1  15  남  덕명중
학생2  17  여  수리중
```

```
# 1-6 행 인덱스/열 이름 변경

import pandas as pd

df = pd.DataFrame([[15,'남','덕명중'], [17,'여','수리중']],
                  index=['준서','예은'],
                  columns=['나이','성별','학교'])

print(df)
print()
print(df.index)
print(df.columns)

df.rename(index={'준서':'학생1', '예은':'학생2'}, inplace=True)
df.rename(columns={'나이':'연령', '성별':'남녀','학교':'소속'}, inplace=True)
# inplace = True 가 반드시 필요하다

print()
print(df)

#### 출력
    나이 성별   학교
준서  15  남  덕명중
예은  17  여  수리중

Index(['준서', '예은'], dtype='object')
Index(['나이', '성별', '학교'], dtype='object')

     연령 남녀   소속
학생1  15  남  덕명중
학생2  17  여  수리중
```

```
# 1-7,8 행, 열 삭제

import pandas as pd

exam_data = {'수학':[90,80,70], '영어':[98,89,95],
             '음악':[85,95,100],'체육':[100,90,90]}

df = pd.DataFrame(exam_data,index=['서준','우현','인아'])
print(df)
print()

# 행 삭제 #
df2 = df.copy()				# df2 = df -> error
df2.drop('우현',inplace=True)
print(df2)
print()

df3 = df.copy()
df3.drop(['우현','인아'],inplace=True)
print(df3)
print()

# 열 삭제 #
df4 = df.copy()
df4.drop('수학',axis=1,inplace=True)
print(df4)

#### 출력
    수학  영어   음악   체육
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90

    수학  영어   음악   체육
서준  90  98   85  100
인아  70  95  100   90

    수학  영어  음악   체육
서준  90  98  85  100

    영어   음악   체육
서준  98   85  100
우현  89   95   90
인아  95  100   90

Process finished with exit code 0

```

파이썬에서는 변수 또는 객체를 복사하려면 .copy()를 사용해야한다.

```
# 1-9 행 선택
import pandas as pd

exam_data = {'수학':[90,80,70], '영어':[98,89,95],
             '음악':[85,95,100],'체육':[100,90,90]}

df = pd.DataFrame(exam_data,index=['서준','우현','인아'])
print(df)
print()

lable1 = df.loc['서준']
position1 = df.iloc[0]
print(lable1)
print()
print(position1)
print()

lable2 = df.loc[['서준','우현']]
position2 = df.iloc[[0,1]]
print(lable2)
print()
print(position2)
print()

lable3 = df.loc['서준':'우현']
position3 = df.iloc[0:1]
print(lable3)
print()
print(position3)

#### 출력
    수학  영어   음악   체육
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90

수학     90
영어     98
음악     85
체육    100
Name: 서준, dtype: int64

수학     90
영어     98
음악     85
체육    100
Name: 서준, dtype: int64

    수학  영어  음악   체육
서준  90  98  85  100
우현  80  89  95   90

    수학  영어  음악   체육
서준  90  98  85  100
우현  80  89  95   90

    수학  영어  음악   체육
서준  90  98  85  100
우현  80  89  95   90

    수학  영어  음악   체육
서준  90  98  85  100
```

.loc or .iloc는 슬라이싱할 때,  값이면 마지막 인덱스 값을 포함한다.

.loc or .iloc는 ()가 아니라 []이다.

```
# 1-10 열 선택

import pandas as pd

exam_data = {'이름':['서준','우현','인아'],
             '수학':[90,80,70],
             '영어':[98,89,95],
             '음악':[85,95,100],
             '체육':[100,90,90]}

df = pd.DataFrame(exam_data)
print(df)
print(type(df))
print()

math1 = df['수학']
print(math1)
print(type(math1))
print()

english = df.영어
print(english)
print(type(english))
print()

music_gym = df[['음악','체육']]
print(music_gym)
print(type(music_gym))
print()

math2 = df[['수학']]
print(math2)
print(type(math2))

#### 출력
   이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90
<class 'pandas.core.frame.DataFrame'>

0    90
1    80
2    70
Name: 수학, dtype: int64
<class 'pandas.core.series.Series'>

0    98
1    89
2    95
Name: 영어, dtype: int64
<class 'pandas.core.series.Series'>

    음악   체육
0   85  100
1   95   90
2  100   90
<class 'pandas.core.frame.DataFrame'>

   수학
0  90
1  80
2  70
<class 'pandas.core.frame.DataFrame'>
```

```
# 1-11 원소 선택

import pandas as pd

exam_data = {'이름':['서준','우현','인아'],
             '수학':[90,80,70],
             '영어':[98,89,95],
             '음악':[85,95,100],
             '체육':[100,90,90]}

df = pd.DataFrame(exam_data)

df.set_index('이름',inplace=True)
print(df)
print()

a = df.loc['서준','음악']
print(a)
print()

b = df.iloc[0,2]
print(b)
print()

c = df.loc['서준',['음악','체육']]
print(c)
print()

d = df.iloc[0,[2,3]]
print(d)
print()

e = df.loc['서준','음악':'체육']
print(e)
print()

f = df.iloc[0,2:]
print(f)
print()

g = df.loc[['서준','우현'],['음악','체육']]
print(g)
print()

h = df.iloc[[0,1],[2,3]]
print(h)
print()

i = df.loc['서준':'우현','음악':'체육']
print(i)
print()

j = df.iloc[0:2,2:]
print(j)

#### 출력
    수학  영어   음악   체육
이름                  
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90

85

85

음악     85
체육    100
Name: 서준, dtype: int64

음악     85
체육    100
Name: 서준, dtype: int64

음악     85
체육    100
Name: 서준, dtype: int64

음악     85
체육    100
Name: 서준, dtype: int64

    음악   체육
이름         
서준  85  100
우현  95   90

    음악   체육
이름         
서준  85  100
우현  95   90

    음악   체육
이름         
서준  85  100
우현  95   90

    음악   체육
이름         
서준  85  100
우현  95   90
```

```
# 1-12 열 추가

import pandas as pd

exam_data = {'이름':['서준','우현','인아'],
             '수학':[90,80,70],
             '영어':[98,89,95],
             '음악':[85,95,100],
             '체육':[100,90,90]}

df = pd.DataFrame(exam_data)
print(df)
print()

df['국어'] = 80
print(df)
print()

#### 출력
   이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90

   이름  수학  영어   음악   체육  국어
0  서준  90  98   85  100  80
1  우현  80  89   95   90  80
2  인아  70  95  100   90  80
```

```
# 1-13 행 추가

import pandas as pd

exam_data = {'이름':['서준','우현','인아'],
             '수학':[90,80,70],
             '영어':[98,89,95],
             '음악':[85,95,100],
             '체육':[100,90,90]}

df = pd.DataFrame(exam_data)
print(df)
print()

df.loc[3] = 0
print(df)
print()

df.loc[4] = ['동규', 90,80,70,60]
print(df)
print()

df.loc['행5'] = df.loc[3]
print(df)

#### 출력
   이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90

   이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90
3   0   0   0    0    0

   이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90
3   0   0   0    0    0
4  동규  90  80   70   60

    이름  수학  영어   음악   체육
0   서준  90  98   85  100
1   우현  80  89   95   90
2   인아  70  95  100   90
3    0   0   0    0    0
4   동규  90  80   70   60
행5   0   0   0    0    0
```

```
# 1-14 원소 값 변경

import pandas as pd

exam_data = {'이름':['서준','우현','인아'],
             '수학':[90,80,70],
             '영어':[98,89,95],
             '음악':[85,95,100],
             '체육':[100,90,90]}

df = pd.DataFrame(exam_data)
print(df)
print()

df.set_index('이름',inplace=True)
print(df)
print()

df.iloc[0][3] = 80
print(df)
print()

df.loc['서준']['체육'] = 90
print(df)
print()

df.loc['서준']['체육'] = 100
print(df)
print()

df.loc[['서준'],['음악','체육']] = 50
print(df)
print()

df.loc['서준',['음악','체육']] = 100,50
print(df)


df.loc['서준']['체육'] = 100
print(df)

#### 출력
   이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90

    수학  영어   음악   체육
이름                  
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90

    수학  영어   음악  체육
이름                 
서준  90  98   85  80
우현  80  89   95  90
인아  70  95  100  90

    수학  영어   음악  체육
이름                 
서준  90  98   85  90
우현  80  89   95  90
인아  70  95  100  90

    수학  영어   음악   체육
이름                  
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90

    수학  영어   음악  체육
이름                 
서준  90  98   50  50
우현  80  89   95  90
인아  70  95  100  90

    수학  영어   음악  체육
이름                 
서준  90  98  100  50
우현  80  89   95  90
인아  70  95  100  90
```

```
# 1-15 행, 열 바꾸기

import pandas as pd

exam_data = {'이름':['서준','우현','인아'],
             '수학':[90,80,70],
             '영어':[98,89,95],
             '음악':[85,95,100],
             '체육':[100,90,90]}

df = pd.DataFrame(exam_data)
print(df)
print()

df = df.transpose()
print(df)
print()

df = df.T
print(df)

#### 출력
   이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90

      0   1    2
이름   서준  우현   인아
수학   90  80   70
영어   98  89   95
음악   85  95  100
체육  100  90   90

   이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90
```

```
# 1-16 특정 열을 행 인덱스로 설정

import pandas as pd

exam_data = {'이름':['서준','우현','인아'],
             '수학':[90,80,70],
             '영어':[98,89,95],
             '음악':[85,95,100],
             '체육':[100,90,90]}

df = pd.DataFrame(exam_data)
print(df)
print()

ndf = df.set_index(['이름'])
print(ndf)
print()

ndf2 = ndf.set_index('음악')
print(ndf2)
print()

ndf3 = ndf.set_index(['수학','음악'])
print(ndf3)

#### 출력
   이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90

    수학  영어   음악   체육
이름                  
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90

     수학  영어   체육
음악              
85   90  98  100
95   80  89   90
100  70  95   90

        영어   체육
수학 음악          
90 85   98  100
80 95   89   90
70 100  95   90
```

```
# 1-17 새로운 배열로 행 인덱스 재지정

import pandas as pd

dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df = pd.DataFrame(dict_data, index=['r0','r1','r2'])
print(df)
print()

new_index = ['r0','r1','r2','r3','r4']
ndf = df.reindex(new_index)
print(ndf)
print()

ndf2 = df.reindex(new_index, fill_value=0)
print(ndf2)

#### 출력
    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15

     c0   c1   c2    c3    c4
r0  1.0  4.0  7.0  10.0  13.0
r1  2.0  5.0  8.0  11.0  14.0
r2  3.0  6.0  9.0  12.0  15.0
r3  NaN  NaN  NaN   NaN   NaN
r4  NaN  NaN  NaN   NaN   NaN

    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15
r3   0   0   0   0   0
r4   0   0   0   0   0
```

```
# 1-18 정수형 위치 인덱스로 초기화

import pandas as pd

dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df = pd.DataFrame(dict_data, index=['r0','r1','r2'])
print(df)
print()

ndf = df.reset_index()
print(ndf)

#### 출력
    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15

  index  c0  c1  c2  c3  c4
0    r0   1   4   7  10  13
1    r1   2   5   8  11  14
2    r2   3   6   9  12  15
```

```
# 1-19 데이터프레임 정렬

import pandas as pd

dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df = pd.DataFrame(dict_data, index=['r0','r1','r2'])
print(df)
print()

ndf = df.sort_index(ascending=False)        # False 역순 정렬
print(ndf)
print()

ndf = df.sort_index(ascending=True)         # True 순행 정렬
print(ndf)

#### 출력
    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15

    c0  c1  c2  c3  c4
r2   3   6   9  12  15
r1   2   5   8  11  14
r0   1   4   7  10  13

    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15
```

```
# 1-20 열 기준 정렬

import pandas as pd

dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df = pd.DataFrame(dict_data, index=['r0','r1','r2'])
print(df)
print()

ndf = df.sort_values(by='c1', ascending=False)
print(ndf)
print()

ndf = df.sort_values(by='c1', ascending=True)
print(ndf)

#### 출력
    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15

    c0  c1  c2  c3  c4
r2   3   6   9  12  15
r1   2   5   8  11  14
r0   1   4   7  10  13

    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15 
```

```
# 1-21 시리즈를 숫자로 나누기

import pandas as pd

student1 = pd.Series({'국어':100, '영어':80, '수학':90})
print(student1)
print()

percentage = student1/200

print(percentage)
print()
print(type(percentage))

#### 출력
국어    100
영어     80
수학     90
dtype: int64

국어    0.50
영어    0.40
수학    0.45
dtype: float64

<class 'pandas.core.series.Series'>
```

```
# 1-22 시리즈 사칙 연산

import pandas as pd

student1 = pd.Series({'국어':100, '영어':80, '수학':90})
student2 = pd.Series({'수학':80, '국어':90, '영어':80})

print(student1)
print()
print(student2)
print()

addition = student1+student2
subtraction = student1-student2
multiplication = student1*student2
division = student1/student2
print(type(division))
print()

result = pd.DataFrame([addition, subtraction, multiplication, division],
                      index=['덧셈', '뺄셈', '곱셈', '나눗셈'])
print(result)

#### 츨력
국어    100
영어     80
수학     90
dtype: int64

수학    80
국어    90
영어    80
dtype: int64

<class 'pandas.core.series.Series'>

              국어        수학      영어
덧셈    190.000000   170.000   160.0
뺄셈     10.000000    10.000     0.0
곱셈   9000.000000  7200.000  6400.0
나눗셈     1.111111     1.125     1.0
```

```
# 1-23 NaN값이 있는 시리즈 연산

import pandas as pd
import numpy as np

student1 = pd.Series({'국어':np.nan, '영어':80, '수학':90})
student2 = pd.Series({'수학':80, '국어':90})

print(student1)
print()
print(student2)
print()

addition = student1+student2
subtraction = student1-student2
multiplication = student1*student2
division = student1/student2
print(type(division))
print()

result = pd.DataFrame([addition, subtraction, multiplication, division],
                      index=['덧셈', '뺄셈', '곱셈', '나눗셈'])
print(result)

#### 출력
국어     NaN
영어    80.0
수학    90.0
dtype: float64

수학    80
국어    90
dtype: int64

<class 'pandas.core.series.Series'>

     국어        수학  영어
덧셈  NaN   170.000 NaN
뺄셈  NaN    10.000 NaN
곱셈  NaN  7200.000 NaN
나눗셈 NaN     1.125 NaN
```

```
# 1-24 연산 메소드 사용 -> 시리즈 연산
import pandas as pd
import numpy as np

student1 = pd.Series({'국어':np.nan, '영어':80, '수학':90})
student2 = pd.Series({'수학':80, '국어':90})

print(student1)
print()
print(student2)
print()

sr_add = student1.add(student2, fill_value=0)
sr_sub = student1.sub(student2, fill_value=0)
sr_mul = student1.mul(student2, fill_value=0)
sr_div = student1.div(student2, fill_value=0)

result = pd.DataFrame([sr_add, sr_sub, sr_mul, sr_div],
                      index=['덧셈', '뺄셈', '곱셈', '나눗셈'])
print(result)

#### 출력
국어     NaN
영어    80.0
수학    90.0
dtype: float64

수학    80
국어    90
dtype: int64

       국어        수학    영어
덧셈   90.0   170.000  80.0
뺄셈  -90.0    10.000  80.0
곱셈    0.0  7200.000   0.0
나눗셈   0.0     1.125   inf
```

NaN : 유효 값이 존재하지 않는 누락 데이터

inf : 무한을 의미

```
# 1-25 데이터 프레임에 숫자 더하기

import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'fare']]
print(df.head())				# 위에서 5행까지만
print()
print(type(df))
print()

addition = df + 10
print(addition.head())
print()
print(type(addition))

#### 출력
    age     fare
0  22.0   7.2500
1  38.0  71.2833
2  26.0   7.9250
3  35.0  53.1000
4  35.0   8.0500

<class 'pandas.core.frame.DataFrame'>

    age     fare
0  32.0  17.2500
1  48.0  81.2833
2  36.0  17.9250
3  45.0  63.1000
4  45.0  18.0500

<class 'pandas.core.frame.DataFrame'>
```

```
# 1-26 데이터프레임끼리 더하기

import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'fare']]
print(df.tail())
print()
print(type(df))
print()

addition = df + 10
print(addition.tail())
print()
print(type(addition))
print()

subtraction = addition - df
print(subtraction.tail())
print()
print(type(subtraction))

#### 출력
      age   fare
886  27.0  13.00
887  19.0  30.00
888   NaN  23.45
889  26.0  30.00
890  32.0   7.75

<class 'pandas.core.frame.DataFrame'>

      age   fare
886  37.0  23.00
887  29.0  40.00
888   NaN  33.45
889  36.0  40.00
890  42.0  17.75

<class 'pandas.core.frame.DataFrame'>

      age  fare
886  10.0  10.0
887  10.0  10.0
888   NaN  10.0
889  10.0  10.0
890  10.0  10.0

<class 'pandas.core.frame.DataFrame'>
```

​	