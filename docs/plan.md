# 계획서


'>' 인용은 인사이트 용으로 사용합니다.
'-' 각 항목별로 나눠줍니다.

## 문제정의 : 
텍스트 데이터를 입력으로 받아 뉴스의 카테고리를 예측하는 딥 러닝 모델을 구현하기

## 시드 고정 :
import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    # 연산 속도보다 재현성을 우선시할 때 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

- random, np.random, torch.manual_seed 은 반드시 고정하기
- Word2Vec, FastText은 임베딩 단계에서 실시

Word2Vec(
    sentences=train_sentences,  # 학습에 쓸 데이터: 토큰화된 문장 리스트 (train만 사용)
    vector_size=128,            # 단어 하나를 몇 차원 벡터로 표현할지 (임베딩 차원)
    window=5,                   # 중심 단어 기준 좌우 몇 단어까지를 "주변"으로 볼지
    min_count=1,                # 최소 등장 횟수. 1이면 한 번만 나온 단어도 모두 포함
    sg=1,                       # 학습 방식: 1이면 skip-gram, 0이면 CBOW
    seed=42,                    # 난수 시드 (초기 벡터 등을 고정해서 재현성 확보)
    workers=1                   # 학습에 쓰는 스레드 수 (1이면 순서가 고정돼 재현성이 좋아짐)
)

## 데이터 수집 : 
18,846개의 뉴스 문서를 20개의 카테고리로 분류한 News Group 20 데이터를 사용
[https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
- data copy 먼저 실시]
- 자료형 확인하기 print(type(data)) => 이거에 따라서 하기 내용 골라서 실시

형태를 먼저 확인한다. 만약 표라면? 
- 어떻게 구성되는지 확인print("크기(행, 열):", data.shape)
- 구성확인 print("열 이름들:", data.columns)
- 몇줄 꺼내 보기print(data.head())

Bunch 형태라면?
- key값있는지 확인하기 print("들어있는 항목들:", data.keys())
- data
- target
- target_names
- DESCR 

List, 배열이라면?
- print("전체 개수:", len(data))
- print("첫 번째 데이터:", data[0])
- print("첫 번째 데이터의 타입:", type(data))

## 데이터 분할 : 홀드 아웃 방식으로 나눈다.
train 6, val 2, test 2
베이스 라인 코드를 사용하되, val 추가하기
- stratify=labels  # 카테고리 비율도 유지시켜주어 사용하기(EDA 이후 추가된 사항)
- train_val_inputs, test_inputs, train_val_targets, test_targets = train_test_split(
		texts, 
		labels, 
		test_size=0.2, 
		random_state=42, 
		stratify=labels
)

train_test_split 은 한번에 이등분만 가능, train을 0.25만큼때서 val 데이터 나누기
- train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    train_val_inputs, 
    train_val_targets, 
    test_size=0.25, 
    random_state=42,
    stratify=train_val_targets
)

## 정제 전, EDA :
EDA는 train 데이터 기준으로 실시한다. 이 결과가 어떤 실험 후보로 이어지나
클래스 분포(labels)
- 카테고리별로 문서 수가 균형 잡혀있는지 확인
- bar plot로 labels 각각 그래프로 출력, 클래스 이름으로 표시

문서 길이 분포
정제를 어떻게 할지 결정하기 전에 전체적인 글의 길이를 확인하는 cell
50% 로 중앙값을 확인하고 나머지는 정제 전 max_len의 길이를 확인해서 정제 후, 다시 문서 길이 분포 확인 예정
- 베이스라인 그대로 사용

단어 빈도
- 제일 많이 나오는 단어
- 워드 클라우드로 시각화
- 1번만 나오는 희귀 단어 확인 및 비율
- 예시 print 하여 확인, 문서 리스트로 뽑아두기

어휘 크기
- 고유 단어 수가 몇개인지 확인하기

노이즈 확인
- 빈 문서, 중복 문서 hash사용해서 확인하기
- train/val/test 복도 확인
- 특수문자, 숫자, 이메일 주소가 얼마나 섞였는지 보기
- 예시 print 하여 확인, 문서 리스트로 뽑아두기

n-gram 분석
- 자주 같이 나오는 단어쌍 확인

데이터 확인 후, 추가 EDA
- alt, comp, misc, rec, sci, soc, talk 라벨 앞에 해당 목록이 보인다. 해당 라벨을 가진 글들을 하나씩 확인해서 어떤 정보를 저렇게 표현했는지 확인한다.
- 공백만 있는 문서와 원본 문서를 1x2로 출력하여 비교한다. 해당 문서가 학습하는데 카테고리 단서를 주는지 못주는지를 확인하고 제외할지 아닐지를 결정한다.

## 정제 클리닝 :
추가 불용어는 train EDA로 정함
- 기본 베이스로 공백이 되어 있기 때문에 해당 내용 사용 후, 피처 엔지니어링에서 변환해도 좋을껏 같다.
- 소문자, 영문자 외 제거, 불용어 제거(특수문자). train/val/test에 같은 clean_text 적용

## 단어 토큰화 : 
- 베이스라인 기본값으로 만들기
train_sentences = [word_tokenize(text) for text in train_inputs]
test_sentences = [word_tokenize(text) for text in test_inputs]
상기에는 val 이 빠져있어
- val_sentences 추가 

## 정제 후, EDA :
상기내용동일

문서 길이 분포
- 베이스라인 그대로 사용
- 'max_len' train 분포로 정할 수 있도록 확인

전처리 전후 비교
- 노이즈 확인 후, 불용어로 판단되는 것은 제거
- 소문자 변환 후, 문서 변화확인
- 예시 print 하여 확인, 문서 리스트로 뽑아두기

## 공통 통일 항목
> 3종 임베딩(Word2Vec, FastText, GloVe)을 공정하게 비교하려면 임베딩 외의 조건은 모두 같아야 한다.
> 값은 모두 **train 기준**으로 정하고, val/test에는 같은 값을 적용만 한다.

**EDA 결과로 정하는 값**
- max_len : **미정** 정제 후 train 문서 길이 분포 확인 후 결정 (베이스라인 280, 95분위수 기준으로 추정)
- 토큰화 방식 : 베이스라인 기본값(word_tokenize) 유지 → `확정`
- 정제 규칙 : 소문자, 영문자 외 제거, 불용어 제거 → train/val/test 동일 적용
- 패딩/OOV 번호 : 베이스라인은 둘 다 0 → 1차 학습은 유지, 분리는 개선 실험 후보

**학습 조건 (1차 학습은 베이스라인 값으로 시작)**
- 시드 : 42 (맨 위 + 모델 생성 직전 셀 첫 줄)
- 배치 크기 : 32 (3종 동일)
- shuffle : train=True, val/test=False
- 학습률 : 0.005 (튜닝 후보)
- 에폭 : 10 (튜닝 후보)
- 분할 : 6:2:2, random_state=42, stratify 사용

**임베딩별로 달라지는 값 (통일 불가, 결과 해석 시 한계로 기록)**
- 임베딩 차원 : 
- 사전과 행렬 : 임베딩마다 별도 (짝으로 생성)
- gensim 설정 : seed=42, workers=1 (Word2Vec, FastText만 해당)
- GloVe : 사전학습 파일 로드 (학습 데이터 규모가 달라 비교 시 주의)

**통일 대상에서 제외**
- 모델 구조 (hidden_dim, output_dim 등) : 별도 단계에서 관리

# 하단 부터는 알고리즘 마다 구성해야해서 상기 공통부분만 전체 통일 시킨다. 단, 모델 구조 제외(hidden_dim, output_dim)

## 백터화 :
정제 전, EDA 확인 후, 적절한 임베딩 방법 결정
- 기본 베이스로 공백 임베딩으로 되어있어 가장 잘 맞는 백터화 선택
- 3종 (사전, 행렬) 만들기. W2V/FastText는 train으로 학습, GloVe는 파일 로드
- gensim seed/workers

## 입력 데이터 구성 : 
데이터 셋
- 베이스라인 기본값으로 고정
- 각각의 함수 정의가 베이스라인에서는 섞여 있기 때문에 셀을 마크다운으로 구분하여 정의한다.

데이터 로더
베이스라인 기본값으로 고정
- batch_size=32, shuffle=True
- train=True, val/test=False, val 로더 추가. Dataset도 train/val/test 3개

## 모델 구현 :
베이스라인의 LSTM 분류기를 이해하고 작성 -> 함수 정의 실시
- class EmbeddingLSTM(nn.Module) 사용해서 작성 실시
- 

## 학습 및 평가
실제로 돌리기가 여기에 포함
- 디스코드 알람설정 실시
- tqdm 포함해서 코드 작성
- seed_everything(42)
- train loss, val loss, val 정확도 기록
- test는 사용하지 않음
- train/evaluate 함수 정의 위치 명시
- val loss 최저 에폭의 가중치 저장
- 임베딩 3종 손실 곡선 비교 그래프
- 학습 후 val 예측값으로 classification_report (클래스별 지표, macro/weighted 평균)

## 계획서 결과 기록 :
### 베이스라인 3종 비교 (val 기준)
> 조건: 시드 42, 배치 32, max_len ○○, 에폭 10, lr 0.005 (3종 동일)

| 임베딩 | 차원 | 사전 크기 | OOV 비율 | 마지막 train loss | val 정확도 | 최소 val loss | 최고 val 정확도 | 최적 에폭 | **EDA이후 확정할 항목 안쓸수도 있음**macro F1 | 메모 | 
|---|---|---|---|---|---|---|---|---|---|---|
| Word2Vec | | | | | | | | | | |
| FastText | | | | | | | | | | |
| GloVe | | | | | | | | | | |

## 실험계획서 :
결과 기록을 가지고 실험할 항목을 나열하고 실험계획서에 맞춰 각 항목에 대해서 작성한다.

## xx차 실험 결과 기록 :
| 항목 | 예시 |
| :--- | :--- |
| **실험 번호** | 0 (베이스라인) |
| **임베딩** | Word2Vec / FastText / GloVe |
| **바꾼 것** | 없음 / lr 0.005 → 0.001 |
| **고정 조건** | 시드 42, 배치 32, max_len 등 |
| **결과** | val 정확도, loss 흐름, (이후) F1 등 |
| **메모** | 느낀 점, 이상한 점 |

## 최종 선발 및 최종 평가 (1차·실험 결과 후 작성): 
test는 여기서 1회만 사용