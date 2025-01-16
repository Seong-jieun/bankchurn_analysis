# 은행 고객 이탈 확률 분석 프로젝트


## 개요
- 주제
    - 은행 고객 이탈 확률 예측(이진분류 분석)
- 배경
    - 은행업계 간 디지털 전환 및 채널 경쟁의 본격화로 자사 고객의 이탈이 심해지고 있음
- 목표
    - 고객 이탈 확률을 예측하고 대응방안을 수립해 고객 이탈을 줄이고자 함
- 캐글 목표
    - Private 10% 이내 (ROC-AUC 기준 363등 점수 : 0.89642)


## 데이터 설명
- 캐글 대회 'Binary Classification with a Bank Churn Dataset'에서 제공하는 데이터를 활용했으며 ABC Multistate bank의 데이터임
- 데이터 볼륨
  - 훈련 데이터 `(165034, 14)`
  - 테스트 데이터 `(110023, 13)`
    
### 데이터 셋 정보
|컬럼명|설명|
|-|-|
|Customer ID|각 고객의 고유 식별자|
|Surname|고객의 성|
|Credit Score|고객의 신용 점수를 나타내는 값|
|Geography|고객이 거주하는 국가(프랑스, 스페인, 독일)|
|Gender|고객의 성별|
|Age|고객의 연령|
|Tenure|고객 은행 이용 기간(년)|
|Balance|고객의 계좌 잔고|
|NumOfProducts|고객이 사용하는 은행 상품 수(예: 저축 계좌, 신용 카드)|
|HasCrCard|고객의 신용 카드 보유 여부(1 = 예, 0 = 아니오)|
|IsActiveMember|고객 활동 여부(1 = 예, 0 = 아니오)|
|EstimatedSalary|고객 추정 급여|
|Exited	|고객 이탈 여부(1 = 예, 0 = 아니오)|

### 사용 라이브러리
|전처리|모델링|시각화|
|-|-|-|
|NumPy|Sklearn|Matplotlib|
|Pandas|AutoML|Seaborn|
||Pycaret||


[자세한 내용보기](https://www.notion.so/96fdf01dd0b94c8dbc07e11e93850aa5)
