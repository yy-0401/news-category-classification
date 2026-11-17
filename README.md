# 20 Newsgroups 뉴스 카테고리 분류: 임베딩 3종 비교

텍스트를 입력으로 받아 20개 뉴스그룹 카테고리를 예측하는 딥러닝 분류 모델을 만들고,
Word2Vec / FastText / GloVe 임베딩을 같은 조건에서 비교합니다.

## 데이터
- scikit-learn `fetch_20newsgroups` (18,846건, 20개 카테고리)
- `remove=('headers', 'footers', 'quotes')` 옵션으로 메타데이터 제거
- Train / Val / Test = 6 : 2 : 2, `stratify` 적용, `random_state=42`

## 설계 원칙
- 분할을 EDA보다 먼저 수행해 데이터 누수를 방지
- 사전, 임베딩, `max_len`은 train 기준으로만 생성하고 val/test에는 적용만
- 임베딩 외 조건은 통일하고, 비교 실험은 한 번에 하나만 변경
- 튜닝과 선택은 val로만 하고, test는 최종 모델에 대해 1회만 사용

## 진행 계획
1. 분할, EDA, 정제, 토큰화
2. 임베딩 3종 베이스라인 학습 및 비교 (val 기준)
3. 최종 모델 선정 및 test 평가

## 결과
(작성 예정)

## 폴더 구조
Mission_10/
├── README.md
├── .gitignore
├── requirements.txt
├── docs/
│   └── plan.md
├── notebooks/
    └── baseline.ipynb


## 문서
- [계획서](docs/plan.md)
- [텍스트 모델 구축 기준서](docs/text_model_playbook.md)

## 출처
- 20 Newsgroups: scikit-learn 문서 및 Jason Rennie의 데이터셋 페이지
- GloVe: Stanford NLP (공식 README의 배포 링크 참고)