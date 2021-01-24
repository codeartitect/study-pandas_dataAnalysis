# PMPDA - Part 4

```
# 4-1,2 선 그래프, 차트 제목, 축 이름 추가
# ,3 한글 폰트 오류 해결
#### 참고
# matplotlib 한글 폰트 오류 문제 해결
from matplotlib import font_manager, rc
font_path = "./malgun.ttf"   #폰트파일의 위치
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
####
# ,4,5,7 그래프 꾸미기, 스타일 서식 지정, Matplotlib 스타일 리스트 출력

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel('C:/Users/User\Project\StudyProject\PMPDA material\part4/시도별 전출입 인구수.xlsx',
                   fillna=0,header=0)

df = df.fillna(method='ffill')

# 서울에서 다른 지역으로 이사를 가게 되는 경우
mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')

df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'],axis=1)
df_seoul.rename({'전입지별':'전입지'},axis=1,inplace=True)
df_seoul.set_index('전입지',inplace=True)

# 서울에서 경기도로 이사가는 경우
sr_one = df_seoul.loc['경기도']

# 그래프 스타일 지정
plt.style.use('ggplot')
# 폰트 설정
plt.rc('font', family='D2Coding')

# 그림 사이즈 지정 (가로 인치, 세로 인치)
plt.figure(figsize=(14,5))

# x축 눈금 라벨 회전
plt.xticks(size=10, rotation='vertical')

# index = 연도, x축 / values = 이동 인구 수, y축 / 마커 스타일/ 마커 사이즈
plt.plot(sr_one.index, sr_one.values, marker='o', markersize=10)


# plt.plot(sr_one)				# 있어도 되고 없어도 된다.


plt.title('서울 -> 경기 인구 이동')
plt.xlabel('기간')
plt.ylabel('이동 인구수')

# 범례 표시
plt.legend(labels=['서울 -> 경기'], loc='best')

# y축 범위 지정
plt.ylim(50000,800000)

# 화살표 표시
plt.annotate('',
             xy=(20, 620000),               # 화살표 머리 부분(끝)
             xytext=(2, 290000),            # 화살표 꼬리 부분(시작)
             xycoords='data',               # 좌표 체계
             arrowprops=dict(arrowstyle='->', color='skyblue', lw=5),   # 화살표 서식
             )

plt.annotate('',
             xy=(47, 450000),
             xytext=(30, 580000),
             xycoords='data',
             arrowprops=dict(arrowstyle='->',color='olive',lw=5),
             )

# 화살표의 주석 표시
plt.annotate('인구 이동 증가(1970-1995)',     # 텍스트 입력
             xy=(10,550000),                  # 텍스트 위치 기준점
             rotation=25,                     # 텍스트 회전 각도
             va='baseline',                   # 텍스트 상하 정렬
             ha='center',                     # 텍스트 좌우 정렬
             fontsize=15,                     # 텍스트 크기
             )

plt.annotate('인구 이동 감소 (1955-2017)',
             xy=(40,560000),
             rotation=-11,
             va='baseline',
             ha='center',
             fontsize=15,
             )

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200122150324428](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200122150324428.png) |

.fillna(method='ffill') :  누락 데이터가 들어 있는 행에 바로 앞에 위치한 행의 데이터 값으로 채운다

mask 자체는 부울 형식이지만 df_seoul 에 들어갈 때는 부울 형식이 아니라 mask의 데이터들이 들어간다

```
# 4-6 Matplotlib 그래프 스타일 출력

import matplotlib.pyplot as plt

print(plt.style.available)

#### 출력
['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']

```

```
# 4-8 Matplotlib 소개

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA material\part4\시도별 전출입 인구수.xlsx',
                   fillna=0, header=0)

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')

df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

sr_one = df_seoul.loc['경기도']

plt.style.use('ggplot')

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(sr_one, 'o', markersize=10)
ax2.plot(sr_one, marker='o', markerfacecolor='green', markersize=10,
         color='olive', linewidth=2, label='서울 -> 경기')
ax2.legend(loc='best')

ax1.set_ylim(50000, 800000)
ax2.set_ylim(50000, 800000)

ax1.set_xticklabels(sr_one.index, rotation=75)
ax2.set_xticklabels(sr_one.index, rotation=75)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200127160751947](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200127160751947.png) |

```
# 4-9 axe 객체 그래프 꾸미기

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA material\part4\시도별 전출입 인구수.xlsx',
                   fillna=0, header=0)

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')

df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

sr_one = df_seoul.loc['경기도']

plt.style.use('ggplot')

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)

ax.plot(sr_one, marker='o', markerfacecolor='orange', markersize=10,
        color='olive', linewidth=2, label='서울 -> 경기')
ax.legend(loc='best')

ax.set_ylim(50000, 800000)

ax.set_title('서울 -> 경기 인구 이동', size=20)

ax.set_xlabel('기간',  size=12)
ax.set_ylabel('이동 인구수', size=12)

ax.set_xticklabels(sr_one.index, rotation=75)

ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| <img src="C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200127162008512.png" alt="image-20200127162008512" style="zoom:150%;" /> |

```
# 4-10 같은 화면에 그래프 추가

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA material\part4\시도별 전출입 인구수.xlsx',
                   fillna=0, header=0)

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')

df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

col_years = list(map(str, range(1970, 2018)))
df_3 = df_seoul.loc[['충청남도', '경상북도', '강원도'], col_years]

plt.style.use('ggplot')

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)

ax.plot(col_years, df_3.loc['충청남도'], marker='o', markerfacecolor='green',
        markersize=10, color='olive', label='서울 -> 충남')
ax.plot(col_years, df_3.loc['경상북도'], marker='o', markerfacecolor='blue',
        markersize=10, color='skyblue', label='서울 -> 경북')
ax.plot(col_years, df_3.loc['강원도'], marker='o', markerfacecolor='red',
        markersize=10, color='magenta', label='서울 -> 강원')

ax.legend(loc='best')

ax.set_title('서울 -> 충남, 경북, 강원 인구 이동', size=20)

ax.set_xlabel('기간', size=12)
ax.set_ylabel('인구 이동수', size=12)

ax.set_xticklabels(col_years, rotation=90)

ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200127163816545](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200127163816545.png) |

```
# 4-11 화면 4분할 그래프

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA material\part4\시도별 전출입 인구수.xlsx',
                   header=0, fillna=0)

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

col_years = list(map(str, range(1970, 2018)))
df_4 = df_seoul.loc[['충청남도','경상북도', '강원도', '전라남도'], col_years]

plt.style.use('ggplot')

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.plot(col_years, df_4.loc['충청남도',:], marker='o', markerfacecolor='green',
         markersize=10, color='olive', linewidth=2, label='서울 -> 충남')
ax2.plot(col_years, df_4.loc['경상북도',:], marker='o', markerfacecolor='blue',
         markersize=10, color='skyblue', linewidth=2, label='서울 -> 경북')
ax3.plot(col_years, df_4.loc['강원도',:], marker='o', markerfacecolor='red',
         markersize=10, color='magenta', linewidth=2, label='사울 -> 강원')
ax4.plot(col_years, df_4.loc['전라남도',:], marker='o', markerfacecolor='orange',
         markersize=10, color='yellow', linewidth=2, label='사울 -> 전남')

ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')
ax4.legend(loc='best')

ax1.set_title('서울 -> 충남 인구 이동', size=15)
ax2.set_title('서울 -> 경북 인구 이동', size=15)
ax3.set_title('서울 -> 강원 인구 이동', size=15)
ax4.set_title('서울 -> 전남 인구 이동', size=15)

ax1.set_xticklabels(col_years, rotation=90)
ax2.set_xticklabels(col_years, rotation=90)
ax3.set_xticklabels(col_years, rotation=90)
ax4.set_xticklabels(col_years, rotation=90)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200131145038632](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200131145038632.png) |

```
# 4-12 matplotlib 스타일 리스트 출력

import matplotlib as mlt

colors = {}

for name, hex in mlt.colors.cnames.items():
    colors[name] = hex

print(colors)

#### 출력
{'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF', 'aquamarine': '#7FFFD4', 'azure': '#F0FFFF', 'beige': '#F5F5DC', 'bisque': '#FFE4C4', 'black': '#000000', 'blanchedalmond': '#FFEBCD', 'blue': '#0000FF', 'blueviolet': '#8A2BE2', 'brown': '#A52A2A', 'burlywood': '#DEB887', 'cadetblue': '#5F9EA0', 'chartreuse': '#7FFF00', 'chocolate': '#D2691E', 'coral': '#FF7F50', 'cornflowerblue': '#6495ED', 'cornsilk': '#FFF8DC', 'crimson': '#DC143C', 'cyan': '#00FFFF', 'darkblue': '#00008B', 'darkcyan': '#008B8B', 'darkgoldenrod': '#B8860B', 'darkgray': '#A9A9A9', 'darkgreen': '#006400', 'darkgrey': '#A9A9A9', 'darkkhaki': '#BDB76B', 'darkmagenta': '#8B008B', 'darkolivegreen': '#556B2F', 'darkorange': '#FF8C00', 'darkorchid': '#9932CC', 'darkred': '#8B0000', 'darksalmon': '#E9967A', 'darkseagreen': '#8FBC8F', 'darkslateblue': '#483D8B', 'darkslategray': '#2F4F4F', 'darkslategrey': '#2F4F4F', 'darkturquoise': '#00CED1', 'darkviolet': '#9400D3', 'deeppink': '#FF1493', 'deepskyblue': '#00BFFF', 'dimgray': '#696969', 'dimgrey': '#696969', 'dodgerblue': '#1E90FF', 'firebrick': '#B22222', 'floralwhite': '#FFFAF0', 'forestgreen': '#228B22', 'fuchsia': '#FF00FF', 'gainsboro': '#DCDCDC', 'ghostwhite': '#F8F8FF', 'gold': '#FFD700', 'goldenrod': '#DAA520', 'gray': '#808080', 'green': '#008000', 'greenyellow': '#ADFF2F', 'grey': '#808080', 'honeydew': '#F0FFF0', 'hotpink': '#FF69B4', 'indianred': '#CD5C5C', 'indigo': '#4B0082', 'ivory': '#FFFFF0', 'khaki': '#F0E68C', 'lavender': '#E6E6FA', 'lavenderblush': '#FFF0F5', 'lawngreen': '#7CFC00', 'lemonchiffon': '#FFFACD', 'lightblue': '#ADD8E6', 'lightcoral': '#F08080', 'lightcyan': '#E0FFFF', 'lightgoldenrodyellow': '#FAFAD2', 'lightgray': '#D3D3D3', 'lightgreen': '#90EE90', 'lightgrey': '#D3D3D3', 'lightpink': '#FFB6C1', 'lightsalmon': '#FFA07A', 'lightseagreen': '#20B2AA', 'lightskyblue': '#87CEFA', 'lightslategray': '#778899', 'lightslategrey': '#778899', 'lightsteelblue': '#B0C4DE', 'lightyellow': '#FFFFE0', 'lime': '#00FF00', 'limegreen': '#32CD32', 'linen': '#FAF0E6', 'magenta': '#FF00FF', 'maroon': '#800000', 'mediumaquamarine': '#66CDAA', 'mediumblue': '#0000CD', 'mediumorchid': '#BA55D3', 'mediumpurple': '#9370DB', 'mediumseagreen': '#3CB371', 'mediumslateblue': '#7B68EE', 'mediumspringgreen': '#00FA9A', 'mediumturquoise': '#48D1CC', 'mediumvioletred': '#C71585', 'midnightblue': '#191970', 'mintcream': '#F5FFFA', 'mistyrose': '#FFE4E1', 'moccasin': '#FFE4B5', 'navajowhite': '#FFDEAD', 'navy': '#000080', 'oldlace': '#FDF5E6', 'olive': '#808000', 'olivedrab': '#6B8E23', 'orange': '#FFA500', 'orangered': '#FF4500', 'orchid': '#DA70D6', 'palegoldenrod': '#EEE8AA', 'palegreen': '#98FB98', 'paleturquoise': '#AFEEEE', 'palevioletred': '#DB7093', 'papayawhip': '#FFEFD5', 'peachpuff': '#FFDAB9', 'peru': '#CD853F', 'pink': '#FFC0CB', 'plum': '#DDA0DD', 'powderblue': '#B0E0E6', 'purple': '#800080', 'rebeccapurple': '#663399', 'red': '#FF0000', 'rosybrown': '#BC8F8F', 'royalblue': '#4169E1', 'saddlebrown': '#8B4513', 'salmon': '#FA8072', 'sandybrown': '#F4A460', 'seagreen': '#2E8B57', 'seashell': '#FFF5EE', 'sienna': '#A0522D', 'silver': '#C0C0C0', 'skyblue': '#87CEEB', 'slateblue': '#6A5ACD', 'slategray': '#708090', 'slategrey': '#708090', 'snow': '#FFFAFA', 'springgreen': '#00FF7F', 'steelblue': '#4682B4', 'tan': '#D2B48C', 'teal': '#008080', 'thistle': '#D8BFD8', 'tomato': '#FF6347', 'turquoise': '#40E0D0', 'violet': '#EE82EE', 'wheat': '#F5DEB3', 'white': '#FFFFFF', 'whitesmoke': '#F5F5F5', 'yellow': '#FFFF00', 'yellowgreen': '#9ACD32'}
```

```
# 4-13 면적 그래프(stacked=False) 그리기

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA material\part4\시도별 전출입 인구수.xlsx',
                   header=0, fillna=0)

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

col_years = list(map(str, range(1970, 2018)))
df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]
df_4 = df_4.transpose()

plt.style.use('ggplot')

df_4.index = df_4.index.map(int)

df_4.plot(kind='area', stacked=False, alpha=0.2, figsize=(20,10))
#		  	    면적	  축적		    

plt.title('서울 -> 타시도 인구 이동', size=30)
plt.ylabel('이동 인구 수', size=20)
plt.xlabel('기간', size=20)
plt.legend(loc='best', fontsize=15)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200131181433787](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200131181433787.png) |

```
# 4-14 면적 그래프(stacked=True) 그리기

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA material\part4\시도별 전출입 인구수.xlsx',
                   header=0, fillna=0)

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

col_years = list(map(str, range(1970, 2018)))
df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]
df_4 = df_4.transpose()

plt.style.use('ggplot')

df_4.index = df_4.index.map(int)

df_4.plot(kind='area', stacked=True, alpha=0.2, figsize=(20,10))

plt.title('서울 -> 타시도 인구 이동', size=30)
plt.ylabel('이동 인구 수', size=20)
plt.xlabel('기간', size=20)
plt.legend(loc='best', fontsize=15)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200201150713089](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200201150713089.png) |

```
# 4-15 axes 객체 속성 변경하기

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA material\part4\시도별 전출입 인구수.xlsx',
                   header=0, fillna=0)

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

col_years = list(map(str, range(1970, 2018)))
df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]
df_4 = df_4.transpose()

plt.style.use('ggplot')

df_4.index = df_4.index.map(int)

ax = df_4.plot(kind='area', stacked=True, alpha=0.2, figsize=(20,10))
print(type(ax))

ax.set_title('서울 -> 타시도 인구 이동', size=30, color='brown', weight='bold')
ax.set_ylabel('이동 인구 수', size=20)
ax.set_xlabel('기간', size=20)
ax.legend(loc='best', fontsize=15)

plt.show()

#### 출력
<class 'matplotlib.axes._subplots.AxesSubplot'>
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200201151440022](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200201151440022.png) |

```
#4-16 세로형 막대 그래프

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA material\part4\시도별 전출입 인구수.xlsx',
                   header=0, fillna=0)

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

col_years = list(map(str, range(2010, 2018)))
df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]
df_4 = df_4.transpose()

plt.style.use('ggplot')

df_4.index = df_4.index.map(int)

df_4.plot(kind='bar', figsize=(20,10), width=0.7,
          color=['orange','green','skyblue','blue'])

plt.title('서울 -> 타시도 인구 이동', size=30)
plt.ylabel('이동 인구 수', size=20)
plt.xlabel('기간', size=20)
plt.ylim(5000, 30000)
plt.legend(loc='best', fontsize=15)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200201152111474](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200201152111474.png) |

```
# 4-17 가로형 막대 그래프

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA material\part4\시도별 전출입 인구수.xlsx',
                   header=0, fillna=0)

df = df.fillna(method='ffill')

mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

col_years = list(map(str, range(2010, 2018)))
df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]

df_4['합계'] = df_4.sum(axis=1)

df_total = df_4[['합계']].sort_values(by='합계', ascending=True)

plt.style.use('ggplot')

df_total.plot(kind='barh', figsize=(10, 5), width=0.5,
          color='cornflowerblue')

plt.title('서울 -> 타시도 인구 이동')
plt.ylabel('전입지')
plt.xlabel('이동 인구 수')

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200201152834788](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200201152834788.png) |

```
# 4-18 2축 그래프 그리기

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

plt.style.use('ggplot')
plt.rcParams['axes.unicode_minus']=False

df = pd.read_excel(r'C:\Users\User\Project\StudyProject\PMPDA\material\part4\남북한발전전력량.xlsx')

# 엑셀 파일의 북한 부분만을 가져온다.
df = df.loc[5:9]
# axis=1 과 axis='columns' 같은 의미를 가진다
df.drop('전력량 (억㎾h)', axis='columns', inplace=True)
df.set_index('발전 전력별', inplace=True)
df = df.T

# df.rename({'합계':'총발전량'}, axis=1, inplace=True)과 같은 의미를 갖는다.
df = df.rename(columns={'합계':'총발전량'})
# shift(1) 은 학칸씩 내리는 것이다.
df['총발전량 - 1년'] = df['총발전량'].shift(1)
df['증감율'] = ((df['총발전량'] / df['총발전량 - 1년']) - 1) * 100

ax1 = df[['수력', '화력']].plot(kind='bar', figsize=(20,10), width=0.7, stacked=True)
# 똑같은(쌍둥이) x축 다른 값을 가진 y축 생성
ax2 = ax1.twinx()
ax2.plot(df.index, df.증감율, ls='--', marker='o', markersize=20,
         color='green', label='전년대비 증감율(%)')

ax1.set_ylim(0, 500)
ax2.set_ylim(-50, 50)

ax1.set_xlabel('연도', size=20)
ax1.set_ylabel('발전량(억㎾h)')
ax2.set_ylabel('전년 대비 증감율(%)')

plt.title('북한 전력 발전량 (1980 ~ 2016)')
# 범례 위치
ax1.legend(loc='upper left')

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200204002023235](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200204002023235.png) |

엑셀 파일을 읽어올 때, loc는 행을 의미하며 헤더 행은 행으로 인식을 하지 않고 1이 아니라 0부터 시작한다.

```
# 4-19 히스토그램

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part4/auto-mpg.csv',
                 header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']

df['mpg'].plot(kind='hist', bins=10, color='coral', figsize=(10, 5))

plt.title('Histogram')
plt.xlabel('mpg')
plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200216021416171](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200216021416171.png) |

```
# 4-20 산점도

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part4/auto-mpg.csv',
                 header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
               'acceleration', 'model year', 'origin', 'name']

df.plot(kind='scatter', x='weight', y='mpg', c='coral', s=10, figsize=(10, 5))
plt.title('Scatter Plot - mpg vs. weight')
plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200216023426583](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200216023426583.png) |

```
# 4-21 버블 차트

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part4/auto-mpg.csv',
                 header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'moder year', 'origin', 'name']

cylinders_size = df.cylinders / df.cylinders.max() * 300

df.plot(kind='scatter', x='weight', y='mpg', c='coral', figsize=(10, 5),
        s=cylinders_size, alpha=0.3)
# alpha는 점의 진하기 정도
plt.title('Scatter Plot : mpg-weight-cylinders')
plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200218210445163](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200218210445163.png) |

```
# 4-22 그림 파일로 저장

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part4/auto-mpg.csv',
                 header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model year', 'origin', 'name']

cylinders_size = df.cylinders / df.cylinders.max() * 300

df.plot(kind='scatter', x='weight', y='mpg', marker='+', figsize=(10, 5),
        cmap='viridis', c=cylinders_size, s=50, alpha=0.3)
# cmap = colormap / c에 변수를 저장하면 값에따라 색이 달라진다.
plt.title('Scatter Plot : mpg-weight-cylinders')

plt.savefig('C:\Project\StudyProject\PMPDA\material\part4/scatter.png')
plt.savefig('C:\Project\StudyProject\PMPDA\material\part4/scatter_transparent.png',
            transparent=True)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200218211549477](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200218211549477.png) |

```
# 4-23 파이 차트

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part4/auto-mpg.csv',
                 header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year', 'origin', 'name']

df['count'] = 1
df_origin = df.groupby('origin').sum()          # 'origin'열 그룹화, 합계 연산
print(df_origin.head())

df_origin.index = ['USA', 'EU', 'JAPAN']

df_origin['count'].plot(kind='pie',
                        figsize=(7, 5),
                        autopct='%1.1f%%',          # 퍼센트 표시
                        startangle=10,              # 시작 각도 - 그래프 사분면 생각하기
                        colors=['chocolate', 'bisque', 'cadetblue'])    # 색상 리스트

plt.title('Model Origin', size=20)
plt.axis('equal')           # 파이 차트의 비율을 같게 -> 타원이 아닌 원에 가깝게 조정
plt.legend(labels=df_origin.index, loc='upper right')
plt.show()

#### 출력
C:\Anaconda3\python.exe C:/Project/StudyProject/test.py
           mpg  cylinders  displacement  ...  acceleration  model_year  count
origin                                   ...                                 
1       5000.8       1556       61229.5  ...        3743.4       18827    249
2       1952.4        291        7640.0  ...        1175.1        5307     70
3       2405.6        324        8114.0  ...        1277.6        6118     79

[3 rows x 7 columns]
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200224185937951](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200224185937951.png) |

```
# 4-24 박스 플롯

import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='D2Coding')

plt.style.use('seaborn-poster')
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('C:\Project\StudyProject\PMPDA\material\part4/auto-mpg.csv',
                 header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model year', 'origin', 'name']

fig = plt.figure(figsize=(15,5))            # 시각화 사이즈 정의
ax1 = fig.add_subplot(1, 2, 1)              # fig에 두 시각화 그래프 위치 정의
ax2 = fig.add_subplot(1, 2, 2)

# 수직 박스 플롯
ax1.boxplot(x=[df[df['origin'] == 1]['mpg'],            # 'origin' 값이 1인 'mpg' 열 데이터 분포
               df[df['origin'] == 2]['mpg'],
               df[df['origin'] == 3]['mpg']],
            labels=['USA', 'EU', 'JAPAN'])

# 수평 박스 플롯
ax2.boxplot(x=[df[df['origin'] == 1]['mpg'],
               df[df['origin'] == 2]['mpg'],
               df[df['origin'] == 3]['mpg']],
            labels=['USA', 'EU', 'JAPAN'],
            vert=False)             # 수평으로 변경

ax1.set_title('제조국가별 연비 분포(수직 박스 플롯)')
ax2.set_title('제조국가별 연비 분포(수평 박스 플롯)')

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200227182052797](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200227182052797.png) |

```
# 4-25 titanic 데이터셋

import seaborn as sns

titanic = sns.load_dataset('titanic')

print(titanic.head())
print()
print(titanic.info())

#### 출력
   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True

[5 rows x 15 columns]

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
```

```
# 4-26 회귀선이 있는 산점도

import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

# 스타일 종류 : darkgrid, whitegrid, dark, white, ticks
sns.set_style('darkgrid')

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

sns.regplot(x='age',
            y='fare',
            data=titanic,
            ax=ax1)         # 그래프 위치 표시

sns.regplot(x='age',
            y='fare',
            data=titanic,
            ax=ax2,
            fit_reg=False)          # 회귀선 없음

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200227183005667](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200227183005667.png) |

```
# 4-27 히스토그램/커널밀도함수

import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

sns.set_style('darkgrid')

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

sns.distplot(titanic['fare'], ax=ax1)                   # 히스토그램, 커널밀도함수 모두 표시
sns.distplot(titanic['fare'], hist=False, ax=ax2)       # 히스토그램 표시x -> 커널밀도함수만 표시
sns.distplot(titanic['fare'], kde=False, ax=ax3)        # 커널밀도함수 표시x -> 히스토그램만 표시

ax1.set_title('titanic fare - hist/ked')
ax2.set_title('titanic fare - ked')
ax3.set_title('titanic fare - hist')

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200227183615864](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200227183615864.png) |

```
# 4-28 히트맵

import matplotlib.pyplot as plt
import seaborn as sns

ttn = sns.load_dataset('titanic')

sns.set_style('darkgrid')

table = ttn.pivot_table(index=['sex'], columns=['class'], aggfunc='size')

sns.heatmap(table,                      # 데이터프레임
            annot=True, fmt='d',        # 데이터 값 표시 여부, 정수형 포맷
            cmap='YlGnBu',              # 컬러맵
            linewidths=.5,              # 구분 선 사이즈
            cbar=False)                 # 컬러 바 표시 여부

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200227184608464](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200227184608464.png) |

```
# 4-29 범주형 데이터의 산점도

import matplotlib.pyplot as plt
import seaborn as sns

ttn = sns.load_dataset('titanic')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

sns.stripplot(x='class',				# 데이터 분산 고려x
              y='age',
              data=ttn,
              ax=ax1)

sns.swarmplot(x='class',				# 데이터 분산 고려o
              y='age',
              data=ttn,
              ax=ax2)

ax1.set_title('Strip Plot')
ax2.set_title('Swarm Plot')

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200228143931461](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200228143931461.png) |

```
# 4-30 막대 그래프

import matplotlib.pyplot as plt
import seaborn as sns

ttn = sns.load_dataset('titanic')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

sns.barplot(x='sex',
            y='survived',
            data=ttn,
            ax=ax1)
sns.barplot(x='sex',
            y='survived',
            hue='class',                # 남성과 여성으로 나누어진 x축을 hue 기준으로 다시 분할(구분)
            data=ttn,
            ax=ax2)
sns.barplot(x='sex',
            y='survived',
            hue='class',
            dodge=False,                # hue로 분할하지만 누적 분할
            data=ttn,
            ax=ax3)

ax1.set_title('titanic survived - sex')
ax2.set_title('titanic survived - sex/class')
ax3.set_title('titanic survived - sex/class(stacked)')

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200228145752433](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200228145752433.png) |

```
# 4-31 빈도 그래프
# 4-30과 y축 값 비교

import matplotlib.pyplot as plt
import seaborn as sns

ttn = sns.load_dataset('titanic')

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

sns.countplot(x='class',
              palette='Set1',
              data=ttn,
              ax=ax1)
sns.countplot(x='class',
              hue='who',
              palette='Set2',
              data=ttn,
              ax=ax2)
sns.countplot(x='class',
              hue='who',
              palette='Set1',
              dodge=False,
              data=ttn,
              ax=ax3)

ax1.set_title('titanic class')
ax2.set_title('titanic class - who')
ax3.set_title('titanic class - who(stacked)')

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200228151453785](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200228151453785.png) |

```
# 4-32 박스 플롯/바이올린 그래프

import matplotlib.pyplot as plt
import seaborn as sns

ttn = sns.load_dataset('titanic')

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

sns.boxplot(x='alive',
            y='age',
            data=ttn,
            ax=ax1)
sns.boxplot(x='alive',
            y='age',
            hue='sex',
            data=ttn,
            ax=ax2)
sns.violinplot(x='alive',
               y='age',
               data=ttn,
               ax=ax3)
sns.violinplot(x='alive',
               y='age',
               hue='sex',
               data=ttn,
               ax=ax4)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200228152122122](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200228152122122.png) |

```
# 4-33 조인트 그래프
# 산점도를 기본으로 표시, x-y 축에 각 변수에 대한 히스토그램 표시
import matplotlib.pyplot as plt
import seaborn as sns

ttn = sns.load_dataset('titanic')

sns.set_style('whitegrid')

j1 = sns.jointplot(x='fare',
                   y='age',
                   data=ttn)
j2 = sns.jointplot(x='fare',
                   y='age',
                   kind='reg',				# 회귀선
                   data=ttn)
j3 = sns.jointplot(x='fare',
                   y='age',
                   kind='hex',				# 육각 산점도
                   data=ttn)
j4 = sns.jointplot(x='fare',
                   y='age',
                   kind='kde',				# 커널 밀집 그래프
                   data=ttn)

j1.fig.suptitle('titanic fare - scatter',  size=15)
j2.fig.suptitle('titanic fare - reg',  size=15)
j3.fig.suptitle('titanic fare - hex',  size=15)
j4.fig.suptitle('titanic fare - kde',  size=15)

plt.show()
```

| 출력                                                         |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20200228154635222](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200228154635222.png) | ![image-20200228154638471](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200228154638471.png) |
| ![image-20200228154643998](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200228154643998.png) | ![image-20200228154647448](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200228154647448.png) |

```
# 4-34 조건에 맞게 화면 분할

import matplotlib.pyplot as plt
import seaborn as sns

ttn = sns.load_dataset('titanic')

sns.set_style('whitegrid')

g = sns.FacetGrid(data=ttn,             # 여러 개의 서브 플롯 생성
                  col='who',            # col = columns 서브 플롯 배열에서 열
                  row='survived')       # row 서브 플롯 배열에서 행

g = g.map(plt.hist, 'age')              # 생성된 서브 플롯에 그래프 객체 전달

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200229050147623](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200229050147623.png) |

```
# 4-35 이변수 데이터 분포

import matplotlib.pyplot as plt
import seaborn as sns

ttn = sns.load_dataset('titanic')

sns.set_style('whitegrid')

ttn_pair = ttn[['age', 'pclass', 'fare']]

g = sns.pairplot(ttn_pair)

plt.show()
```

| 출력                                                         |
| ------------------------------------------------------------ |
| ![image-20200229155345275](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20200229155345275.png) |

```
# 4-36~37 지도 만들기, 지도 스타일 적용

import folium

seoul_map = folium.Map(location=[37.55, 126.98], zoom_start=12)
seoul_map2 = folium.Map(location=[37.55, 126.98], zoom_start=12, tiles='Stamen Terrain')
seoul_map3 = folium.Map(location=[37.55, 126.98], zoom_start=15, tiles='Stamen Toner')

seoul_map.save('C:\Project\StudyProject\PMPDA\material\part4/seoul.html')
seoul_map2.save('C:\Project\StudyProject\PMPDA\material\part4/seoul2.html')
seoul_map3.save('C:\Project\StudyProject\PMPDA\material\part4/seoul3.html')
```

```
# 4-38 지도에 마커 표시하기

import pandas as pd
import folium

df = pd.read_excel('C:\Project\StudyProject\PMPDA\material\part4/서울지역 대학교 위치.xlsx')
df.index = df['Unnamed: 0']
# 엑셀 파일에서 columns 첫 번째 값의 이름이 정해져 있지 않으므로 인덱스 설정을 저렇게 해줘야한다.
# 그리고 인덱스 설정(df.index 값을 설정)을 해줘야 마커 popup에 이름이 떠오른다.
# 밑에 반복문에서 사용하기 때문에 꼭 설정!!

seoul_map = folium.Map(location=[37.55, 126.98], tiles='Stamen Terrain',        # tiles : 맵 스타일
                       zoom_start=12)

for name, lat, lng in zip(df.index, df['위도'], df['경도']):

    folium.Marker([lat, lng], popup=name).add_to(seoul_map)  
    # popup : 마커 클릭시 팝업창에 표시해주는 텍스트
    
seoul_map.save('C:\Project\StudyProject\PMPDA\material\part4/seoul_colleges.html')
```

```
# 4-39 지도에 원형 마커 표시

import pandas as pd
import folium

df = pd.read_excel('C:\Project\StudyProject\PMPDA\material\part4/서울지역 대학교 위치.xlsx')
df.index = df['Unnamed: 0']

seoul_map = folium.Map(location=[37.55, 126.98], tiles='Stamen Terrain',        # tiles : 맵 스타일
                       zoom_start=12)

for name, lat, lng in zip(df.index, df['위도'], df['경도']):

    folium.CircleMarker([lat, lng],
                  popup=name,
                  radius=10,
                  color='brown',		# 원 둘레 색
                  fill=True,
                  fill_color='coral',	# 원 넓이 색
                  fill_opacity=0.7		# 투명도
                  ).add_to(seoul_map)  # popup : 마커 클릭시 팝업창에 표시해주는 텍스트

seoul_map.save('C:\Project\StudyProject\PMPDA\material\part4/seoul_colleges2.html')
```

```
# 4-40 지도 영역에 단계구분도 표시하기

import pandas as pd
import folium
import json

file_path = 'C:\Project\StudyProject\PMPDA\material\part4\경기도인구데이터.xlsx'
df = pd.read_excel(file_path, index_col='구분')
df.columns = df.columns.map(str)

geo_path = 'C:\Project\StudyProject\PMPDA\material\part4\경기도행정구역경계.json'

try:
    geo_data = json.load(open(geo_path, encoding='utf_8'))
except:
    geo_data = json.load(open(geo_path, encoding='utf_8-sig'))

g_map = folium.Map(location=[37.5502, 126.982],
                   tiles='Stamen Terrain', zoom_start=9)

# year = str(input('연도를 입력하세요 : '))
year = '2017'

folium.Choropleth(geo_data= geo_data,
                  data= df[year],                   # 표시하려는 데이터
                  columns= [df.index, df[year]],    # 열 지정
                  fill_color= 'YlOrRd',
                  fill_opacity= 0.7,
                  line_opacity= 0.3,
                  threshold_scale= [10000, 100000, 300000, 500000, 700000],
                  key_on= 'feature.properties.name',
                  ).add_to(g_map)

g_map.save('C:\Project\StudyProject\PMPDA\material\part4\gyonggi_population_{}.html'.format(year))
```


