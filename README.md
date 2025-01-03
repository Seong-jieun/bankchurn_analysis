# 은행 고객 이탈 확률 분석 프로젝트
- 멋쟁이사자처럼 데이터분석스쿨 2기 파이널 프로젝트
- 2인 프로젝트

## 목차


## 개요
- 주제
    - 은행 고객 이탈 확률 예측(이진분류 분석)
- 배경
    - 은행업계 간 디지털 전환 및 채널 경쟁의 본격화로 자사 고객의 이탈이 심해지고 있다
- 목표
    - 고객 이탈 확률을 예측하고 대응방안을 수립해 고객 이탈을 줄이고자 한다
- 캐글 목표
    - Private 10% 이내 (ROC-AUC 기준 363등 점수 : 0.89642)
- 🙋‍♀️ 역할
  - 데이터 전처리 및 EDA
  - 머신러닝 모델링
  - 인사이트 도출
  - 기획서 및 보고서 작성


## 데이터 설명
- 캐글 대회 'Binary Classification with a Bank Churn Dataset'에서 제공하는 데이터를 활용했으며 ABC Multistate bank의 데이터이다
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
|numpy|sklearn|matplotlib|
|pandas|AutoML|seaborn|
||pycaret||


## 분석 내용
✔️**EDA**
- 범주형 피처와 이탈률 비교
  |<img src="https://github.com/user-attachments/assets/ccdebbc4-8b5e-4852-9f68-8b9ea6376044">|<img src="https://github.com/user-attachments/assets/411206a3-35b9-4966-9885-4611113f4b8e">|<img src="https://github.com/user-attachments/assets/017dbfb0-9e4b-430d-9459-e6c5060fe1db">|
  |-|-|-|
  |<img src="https://github.com/user-attachments/assets/e2caa231-9d93-41dd-ad34-ff7203367b95">|<img src="https://github.com/user-attachments/assets/3e31cc74-3d57-452e-8d88-6e33497e7d54">|<img src="https://github.com/user-attachments/assets/5994c192-6a97-4a6d-bd3a-3ed54971e435">|

- 수치형 피처와 이탈률 비교
  |<img src="https://github.com/user-attachments/assets/1249308d-ee1d-474a-9ce6-643f362b2874">|<img src="https://github.com/user-attachments/assets/2f0e4eb5-e076-48bc-891b-21077b361b32">|
  |-|-|
  |<img src="https://github.com/user-attachments/assets/1c3662ab-db10-4907-9d28-392622b2953a">|<img src="https://github.com/user-attachments/assets/9bbd06c9-40ed-414b-bb4a-d98ba62c9ab0">|

- 이상치 확인
  |<img src="https://github.com/user-attachments/assets/7e9d2ef7-ac9d-4426-b350-fc16b8b73389">|<img src="https://github.com/user-attachments/assets/b2cffffd-13c6-4c02-9a40-90fecaad81d3">|
  |-|-|

✔️**데이터 전처리**
- MinMax Scaling 진행
    - 대상피처: CreditScore, Age, Balance, EstimatedSalary
- 파생변수 생성
    - IsSenior: 60세 이상 고객인지 여부
    - IsActive_by_CreditCard: 고객활동여부와 신용카드 보유 고객 간의 관계
    - Products_Per_Tenure: 이용 상품 수 대비 이용기간(년)
    - AgeCategory: 연령을 20년 기준으로 나눈 피처
    - Sur_Geo_Gend_Sal: 범주형 피처들(Surname, Geography, Gender)과 EstimatedSalary피처를 string으로 변환 후 결합하여 생성
- Label Encoding 진행
    - 대상피처: Geography, Gender, IsSenior, IsActive_by_CreditCard, Products_Per_Tenure, AgeCategory
- TfidfVectorizer를 사용하여 벡터화
    - Surname피처를 벡터화 후 SVD를 사용하여 차원 축소
    - 벡터화 된 Surname 피처의 각 특징에 대한 피처 생성
- CatBoost Encoding 진행
    - 대상피처: Sur_Geo_Gend_Sal
  
✔️**가설 설정 및 검증**
- 가설 1. 고객 이탈 현상은 고객의 연령대와 상관관계가 있을 것이다
    - 검증
      - 고객 이탈 현상과 고객 연령은 약한 양의 상관관계가 있다
      - 이탈 하지 않은 고객은 20대 후반-30대에, 이탈한 고객은 30대 후반-50대 중반에 많이 분포 한다
- 가설 2. 고객 이탈 현상과 신용 점수, 계좌 잔고, 급여는 밀접한 연관이 있을 것이다
    - 검증
      - 고객 이탈 현상과 신용점수는 음의 상관관계, 계좌잔고는 약한 양의 상관관계, 급여는 아주 약한 양의 상관관계를 가진다
- 가설 3. 은행 이용 기간이 길수록 이탈율이 낮을 것이다
    - 검증
      - 고객 이탈 현상과 은행 이용 기간과는 음의 상관관계를 가진다

✔️**진행 방식**
- 알고리즘 선정을 위해 6개 알고리즘을 학습 후 비교
- AutoML을 이용하여 Accuracy 기준 상위 3개 모델 확인
- GridSearchCV를 이용하여 교차검증 및 하이퍼파라미터 튜닝 진행
- Feature Importance에 따라 중요도가 낮은 피처 삭제 후 모델링 반복
- 모델 적용 및 예측
- 캐글 제출
  
✔️**모델 선정**
- 알고리즘 6개를 비교하여 AUC값이 높게 나온 모델(LGBM)을 선정했다
  |알고리즘|AUC||알고리즘|AUC|
  | --- | --- | --- | --- | --- |
  | Logistic Regression | 0.60545 || XGBoost | 0.87444 |
  | Decision Tree | 0.86804 || **LGBM** | **0.87601** |
  | Random Forest | 0.84834 || CatBoost | 0.87560 |
- 마지막 모델링(5차)에서는 1차 AutoML시 상위 3개의 모델 중 AUC값이 제일 높았던 GBC를 사용했다
  <img src="https://github.com/user-attachments/assets/1f9cedcd-089b-4933-bcc7-598bdd621b77" width="70%" height="70%">
  
✔️**모델링 결과**
|  | 1차 | 2차 | 3차 | 4차 | 5차 |
| --- | --- | --- | --- | --- | --- |
| 캐글 점수 | 0.79369 | 0.88828 | 0.87956 | 0.87961 | 0.89191 |
| AUC | 0.8765 | 0.8913 | 0.8930 | 0.8920 | 0.8920 |

- 1차 모델링
  - 파생 변수 생성 전, 가설 검증에 필요하지 않다고 판단한 피처(Surname, Geography, Gender) 삭제 후 LGBM 모델링 진행
- 2차 모델링
  - 삭제했던 Geography, Gender 피처를 추가하여 OneHot Encoding한 후 LGBM 모델링 진행
- 3차 모델링
  - 파생 변수 생성 후 LGBM 모델링 진행
  - 파생변수 생성 후 상관계수 확인
     <img src="https://github.com/user-attachments/assets/11609e0b-05f3-48f6-827b-ec45087f2f71" width="70%" height="70%">
- 4차 모델링
  - Feature Importance가 낮은 피처들을 삭제 후 LGBM 모델링 진행
  - 캐글 점수가 약간 오르고 AUC값은 조금 떨어졌지만 그 차이가 적다
    <img src="https://github.com/user-attachments/assets/5eb8b36e-e770-4f8a-b744-33be035cbbbc" withd="70%" height="70%">
- 5차 모델링
  - AutoML결과 AUC가 제일 높았던 GBC 모델을 사용하여 모델링 진행
  - 캐글점수가 올랐으며 AUC값은 변동이 없다
  - AUC값이 같은 것으로 보아 LGBM과 GBC 모델의 성능은 크게 차이 나지 않은 것으로 보인다

- 최종 캐글 점수: 0.89191 (목표점수였던 0.89642는 달성하지 못하였다)
- 최종 AUC: 0.8920
- 하이퍼 파라미터 튜닝 결과 LGBM보다 GBC를 사용한 경우에 캐글 점수와 AUC값이 더 높게 나왔다
- 두 모델 사이에 Accuracy와  AUC값에 차이가 있었던 원인을 GBC가 오버피팅 관리가 더 잘 되었다고 분석하였다

✔️**인사이트 제안**
1. 1개 혹은 2개의 핵심 상품 개발
2. 개발한 핵심 상품을 포함한 여러 개의 기존 상품 이용 시 부가 혜택 제공
3. 20대 후반~30대 고객 맞춤 상품 개발
4. 40대 이상 고객 맞춤 상품 개발


## 회고 및 개선점
### 회고
- 프로젝트 진행 초반에 가설검증에 초점을 맞추느라 데이터 전처리 과정에서 피처를 임의로 삭제한 후 모델링을 진행했던 것이 프로젝트 진행 속도를 늦췄던 것 같다
- 고객의 계좌잔고가 0인 경우가 많았는데 이를 활용하지 못하였다
- 목표했던 캐글 점수를 달성하지 못하였다
  
### 개선점
- 모델링에 앞서 raw데이터의 모든 피처 특성을 살펴보는데 시간을 더 투자한 후 진행하면 데이터에 대한 이해도와 모델링 성능을 더 높일 수 있을 것이다
- 각 모델의 하이퍼파라미터를 더 깊게 탐색하여 최적의 조합을 찾을 수 있을 것이다
- LGBM보다 GBC에서 더 나은 결과가 나올 수 있었던 이유를 오버피팅에서 찾았으므로 오버피팅 방지를 모델링 단계에서 진행하면 다른 결과를 얻을 수 있을 것으로 보인다
- 계좌잔고 0값에 대한 처리를 하여 모델링에 반영한다
