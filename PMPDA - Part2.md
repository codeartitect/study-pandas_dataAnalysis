# PMPDA - Part2

```
# 2-1 CSV 파일 찾기

import pandas as pd

file_path = 'E:/Project/StudyProject/PMPDA/material/part2/read_csv_sample.csv'

df1 = pd.read_csv(file_path)
print(df1)
print()

df2 = pd.read_csv(file_path, header=None)
print(df2)
print()

df3 = pd.read_csv(file_path, index_col=None)
print(df3)
print()

df4 = pd.read_csv(file_path, index_col='c0')
print(df4)

#### 출력
   c0  c1  c2  c3
0   0   1   4   7
1   1   2   5   8
2   2   3   6   9

    0   1   2   3
0  c0  c1  c2  c3
1   0   1   4   7
2   1   2   5   8
3   2   3   6   9

   c0  c1  c2  c3
0   0   1   4   7
1   1   2   5   8
2   2   3   6   9

    c1  c2  c3
c0            
0    1   4   7
1    2   5   8
2    3   6   9
```

```
# 2-2 Excel 파일 읽기

import pandas as pd

df1 = pd.read_excel('E:/Project/StudyProject/PMPDA/material/part2/남북한발전전력량.xlsx')
df2 = pd.read_excel('E:/Project/StudyProject/PMPDA/material/part2/남북한발전전력량.xlsx', header=None)

print(df1)
print()
print(df2)

#### 출력
  전력량 (억㎾h) 발전 전력별  1990  1991  1992  1993  ...  2011  2012  2013  2014  2015  2016
0        남한     합계  1077  1186  1310  1444  ...  4969  5096  5171  5220  5281  5404
1       NaN     수력    64    51    49    60  ...    78    77    84    78    58    66
2       NaN     화력   484   573   696   803  ...  3343  3430  3581  3427  3402  3523
3       NaN    원자력   529   563   565   581  ...  1547  1503  1388  1564  1648  1620
4       NaN    신재생     -     -     -     -  ...     -    86   118   151   173   195
5        북한     합계   277   263   247   221  ...   211   215   221   216   190   239
6       NaN     수력   156   150   142   133  ...   132   135   139   130   100   128
7       NaN     화력   121   113   105    88  ...    79    80    82    86    90   111
8       NaN    원자력     -     -     -     -  ...     -     -     -     -     -     -

[9 rows x 29 columns]

          0       1     2     3     4   ...    24    25    26    27    28
0  전력량 (억㎾h)  발전 전력별  1990  1991  1992  ...  2012  2013  2014  2015  2016
1         남한      합계  1077  1186  1310  ...  5096  5171  5220  5281  5404
2        NaN      수력    64    51    49  ...    77    84    78    58    66
3        NaN      화력   484   573   696  ...  3430  3581  3427  3402  3523
4        NaN     원자력   529   563   565  ...  1503  1388  1564  1648  1620
5        NaN     신재생     -     -     -  ...    86   118   151   173   195
6         북한      합계   277   263   247  ...   215   221   216   190   239
7        NaN      수력   156   150   142  ...   135   139   130   100   128
8        NaN      화력   121   113   105  ...    80    82    86    90   111
9        NaN     원자력     -     -     -  ...     -     -     -     -     -

[10 rows x 29 columns]
```

```
# 2-3 JSON 파일 읽기

import pandas as pd

df = pd.read_json('E:/Project/StudyProject/PMPDA/material/part2/read_json_sample.json')
print(df)
print()
print(df.index)

#### 출력
           name  year        developer opensource
pandas           2008    Wes Mckinneye       True
NumPy            2006  Travis Oliphant       True
matplotlib       2003   John D. Hunter       True

Index(['pandas', 'NumPy', 'matplotlib'], dtype='object')
```

```
# 2-4 웹에서 표 정보 읽기

import pandas as pd

url = 'E:\Project\StudyProject\PMPDA\material\part2\sample.html'

tables = pd.read_html(url)

print(len(tables))
print()

for i in range(len(tables)):
    print("tables[{}]".format(i))
    print(tables[i])
    print()

df = tables[1]

df.set_index(['name'], inplace=True)
print(df)

#### 출력
2

tables[0]
   Unnamed: 0  c0  c1  c2  c3
0           0   0   1   4   7
1           1   1   2   5   8
2           2   2   3   6   9

tables[1]
         name  year        developer  opensource
0       NumPy  2006  Travis Oliphant        True
1  matplotlib  2003   John D. Hunter        True
2      pandas  2008    Wes Mckinneye        True

            year        developer  opensource
name                                         
NumPy       2006  Travis Oliphant        True
matplotlib  2003   John D. Hunter        True
pandas      2008    Wes Mckinneye        True
```



정규표현식 공부 후 할 것

```
# 2-5 미국 ETF 리스트 가져오기
```





```
# 2-6 구글 지오코딩 위치 정보

import googlemaps
import pandas as pd

my_keys = 'AIzaSyDJIUS4brLelVttRAOYVPaonQ19uyvDqnE'

maps = googlemaps.Client(key=my_keys)

lat = []        # 위도
lng = []        # 경도

places = ['서울시청', '국립국악원', '해운대해수욕장']

i = 0

for place in places:
    i = i+1
    try:
        print(i, place)
        geo_location = maps.geocode(place)[0].get('geometry')
        lat.append(geo_location['location']['lat'])
        lng.append(geo_location['location']['lng'])

    except:
        lat.append('')
        lng.append('')

df = pd.DataFrame({'위도':lat, '경도':lng}, index=places)
print()
print(df)

#### 출력
1 서울시청
2 국립국악원
3 해운대해수욕장

                위도          경도
서울시청     37.566295  126.977945
국립국악원    37.477759  127.008304
해운대해수욕장  35.158698  129.160384
```

구글 맵 api key 인증번호

AIzaSyDJIUS4brLelVttRAOYVPaonQ19uyvDqnE

```
# 2-7,8,9 CSV, JSON, Exel 파일로 저장

import pandas as pd

data = {'name':['Jerry','Riah','Paul'],
        'algol':['A','A+','B'],
        'basic':['C','B','B+'],
        'c++':['B+','C','C+']
        }

df = pd.DataFrame(data)
df.set_index('name',inplace=True)
print(df)

df.to_csv('E:\Project\StudyProject\PMPDA material\part2/df_sample.csv')
df.to_csv('E:\Project\StudyProject\PMPDA material\part2/df_sample.json')
df.to_csv('E:\Project\StudyProject\PMPDA material\part2/df_sample.xlsx')

#### 출력
      algol basic c++
name                 
Jerry     A     C  B+
Riah     A+     B   C
Paul      B    B+  C+
####
df_sample.csv 파일 -> 지정 경로에 생성
df_sample.json 파일 -> 지정 경로에 생성
df_sample.xlsx 파일 -> 지정 경로에 생성
```

```
# 2-10 ExelWriter() 활용

import pandas as pd

data1 = {'name':['Jerry','Riah','Paul'],
         'algol':['A','A+','B'],
         'basic':['C','B','B+'],
         'c++':['B+','C','C+']
        }

data2 = {'c0':[1,2,3],
         'c1':[4,5,6,],
         'c2':[7,8,9],
         'c3':[10,11,12],
         'c4':[13,14,15]
        }

df1 = pd.DataFrame(data1)
df1.set_index('name',inplace=True)

df2 = pd.DataFrame(data2)
df2.set_index('c0',inplace=True)

print(df1)
print()

print(df2)
print()

writer = pd.ExcelWriter('E:\Project\StudyProject\PMPDA material\part2/df_excelwriter.xlsx')
df1.to_excel(writer, sheet_name='sheet1')
df2.to_excel(writer, sheet_name='sheet2')
writer.save()

#### 출력
      algol basic c++
name                 
Jerry     A     C  B+
Riah     A+     B   C
Paul      B    B+  C+

    c1  c2  c3  c4
c0                
1    4   7  10  13
2    5   8  11  14
3    6   9  12  15
####
df_excelwriter.xlsx 파일에 sheet1 과 sheet2 생성 저장
```
