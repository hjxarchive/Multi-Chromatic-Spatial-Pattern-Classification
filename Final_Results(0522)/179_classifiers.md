# 179 Classifiers — "Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?"
> Fernández-Delgado et al., JMLR 15 (2014) 3133–3181

구현 표기: **C** = C/C++, **m** = Matlab, **R** = R 패키지 직접 사용, **t** = R + caret, **w** = Weka

---

## 1. Discriminant Analysis (DA) — 20개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 1 | lda | R | Linear Discriminant Analysis (MASS 패키지) |
| 2 | lda2 | t | LDA, 유지할 컴포넌트 수 튜닝 (MASS) |
| 3 | rrlda | R | Robust Regularized LDA (rrlda 패키지) |
| 4 | sda | t | Shrinkage Discriminant Analysis + CAT score 변수 선택 (sda 패키지) |
| 5 | slda | t | Left-Spherically Distributed Linear Scores 기반 LDA (ipred 패키지) |
| 6 | stepLDA | t | Forward/Backward 변수 선택 LDA (klaR 패키지) |
| 7 | sddaLDA | R | Stepwise Diagonal Discriminant Analysis — LDA (SDDA 패키지) |
| 8 | PenalizedLDA | t | Penalized LDA, lasso/fused 계수 사용 (penalizedLDA 패키지) |
| 9 | sparseLDA | R | Sparse LDA, SDA criterion 최소화 (sparseLDA 패키지) |
| 10 | qda | t | Quadratic Discriminant Analysis (MASS 패키지) |
| 11 | QdaCov | t | Robust QDA (rrcov 패키지) |
| 12 | sddaQDA | R | Stepwise Diagonal Discriminant Analysis — QDA (SDDA 패키지) |
| 13 | stepQDA | t | Forward/Backward 변수 선택 QDA (klaR 패키지) |
| 14 | fda | R | Flexible Discriminant Analysis, 선형 회귀 (mda 패키지) |
| 15 | fda | t | FDA + nprune 파라미터 튜닝 (mda 패키지) |
| 16 | mda | R | Mixture Discriminant Analysis (mda 패키지) |
| 17 | mda | t | MDA + subclasses 파라미터 튜닝 (mda 패키지) |
| 18 | pda | t | Penalized Discriminant Analysis, lambda 튜닝 (mda 패키지) |
| 19 | rda | R | Regularized Discriminant Analysis (klaR 패키지) |
| 20 | hdda | R | High-Dimensional Discriminant Analysis (HDclassif 패키지) |

---

## 2. Bayesian (BY) — 6개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 21 | naiveBayes | R | Naive Bayes, Gaussian 커널 (klaR 패키지) |
| 22 | vbmpRadial | t | Variational Bayesian Multinomial Probit Regression, RBF 커널 (vbmp 패키지) |
| 23 | NaiveBayes | w | Naive Bayes (Weka) |
| 24 | NaiveBayesUpdateable | w | 반복 업데이트 Naive Bayes (Weka) |
| 25 | BayesNet | w | Bayesian Network, K2 탐색 + simpleEstimator (Weka) |
| 26 | NaiveBayesSimple | w | 단순 Naive Bayes, 정규분포 수치 특성 (Weka) |

---

## 3. Neural Networks (NNET) — 21개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 27 | rbf | m | Radial Basis Function 신경망, Gaussian spread 튜닝 (Matlab newrb) |
| 28 | rbf | t | RBF 네트워크, 은닉 뉴런 수 튜닝 (RSNNS 패키지) |
| 29 | RBFNetwork | w | K-means 센터 선택 + 선형 회귀 분류 (Weka) |
| 30 | rbfDDA | t | RBF with Dynamic Decay Adjustment (RSNNS 패키지) |
| 31 | mlp | m | Multi-Layer Perceptron, 은닉 뉴런 수 튜닝 (Matlab newpr) |
| 32 | mlp | C | MLP, 학습 알고리즘 + 은닉 뉴런 수 튜닝 (FANN 라이브러리) |
| 33 | mlp | t | MLP, 네트워크 크기 튜닝 (RSNNS 패키지) |
| 34 | avNNet | t | 5개 MLP 앙상블, size·weight decay 튜닝 (caret 패키지) |
| 35 | mlpWeightDecay | t | MLP + weight decay 튜닝 (RSNNS 패키지) |
| 36 | nnet | t | MLP, size·weight decay 튜닝 (nnet 패키지) |
| 37 | pcaNNet | t | PCA 전처리 후 MLP 학습 (nnet 패키지) |
| 38 | MultilayerPerceptron | w | MLP, sigmoid 은닉층 + 선형 출력 (Weka) |
| 39 | pnn | m | Probabilistic Neural Network, Gaussian spread 튜닝 (Matlab newpnn) |
| 40 | elm | m | Extreme Learning Machine, 6종 활성함수 + 은닉 뉴런 수 튜닝 (Matlab) |
| 41 | elm kernel | m | ELM with Gaussian Kernel, 정규화 파라미터 + kernel spread 튜닝 (Matlab) |
| 42 | cascor | C | Cascade Correlation Neural Network (FANN 라이브러리) |
| 43 | lvq | R | Learning Vector Quantization (class 패키지) |
| 44 | lvq | t | LVQ, size·k 파라미터 튜닝 (class 패키지) |
| 45 | bdk | R | Bi-Directional Kohonen Map (kohonen 패키지) |
| 46 | dkp | C | Direct Kernel Perceptron, kernel spread 튜닝 (저자 구현) |
| 47 | dpp | C | Direct Parallel Perceptron, n=3 퍼셉트론 (저자 구현) |

---

## 4. Support Vector Machines (SVM) — 10개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 48 | svm | C | SVM, Gaussian 커널, C·gamma 튜닝 (LibSVM) |
| 49 | svmlight | C | SVM, Gaussian 커널 (SVMlight) |
| 50 | LibSVM | w | SVM, Gaussian 커널 (LibSVM via Weka) |
| 51 | LibLINEAR | w | 대규모 선형 분류 (LibLINEAR via Weka) |
| 52 | svmRadial | t | SVM, Gaussian 커널, C·spread 튜닝 (kernlab 패키지) |
| 53 | svmRadialCost | t | SVM, Gaussian 커널, C만 튜닝 (kernlab 패키지) |
| 54 | svmLinear | t | SVM, 선형 커널, C 튜닝 (kernlab 패키지) |
| 55 | svmPoly | t | SVM, 다항 커널 (1·2·3차), scale·offset·C 튜닝 (kernlab 패키지) |
| 56 | lssvmRadial | t | Least Squares SVM, Gaussian 커널 (kernlab 패키지) |
| 57 | SMO | w | SVM, Sequential Minimal Optimization, 이차 커널 (Weka) |

---

## 5. Decision Trees (DT) — 14개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 58 | rpart | R | Recursive Partitioning (rpart 패키지) |
| 59 | rpart | t | rpart + complexity parameter 튜닝 (rpart 패키지) |
| 60 | rpart2 | t | rpart + 트리 깊이 튜닝 (rpart 패키지) |
| 61 | obliqueTree | R | 경사(Oblique) 분할 이진 재귀 분할 (oblique.tree 패키지) |
| 62 | C5.0Tree | t | C5.0 단일 결정 트리 (C50 패키지) |
| 63 | ctree | t | Conditional Inference Tree, mincriterion 튜닝 (party 패키지) |
| 64 | ctree2 | t | ctree + 최대 트리 깊이 튜닝 (party 패키지) |
| 65 | J48 | w | Pruned C4.5 Decision Tree (Weka) |
| 66 | J48 | t | J48 pruned/unpruned C5.0 트리 (RWeka 패키지) |
| 67 | RandomSubSpace | w | 여러 REPTree + 랜덤 입력 부분집합 (Weka) |
| 68 | NBTree | w | 리프에 Naive Bayes를 가진 결정 트리 (Weka) |
| 69 | RandomTree | w | 비가지치기 트리, 랜덤 입력 테스트 (Weka) |
| 70 | REPTree | w | Reduced Error Pruning 결정 트리 (Weka) |
| 71 | DecisionStump | w | 단일 노드 결정 트리 (Weka) |

---

## 6. Rule-Based Classifiers (RL) — 12개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 72 | PART | w | Partial C4.5 트리 기반 규칙 생성 (Weka) |
| 73 | PART | t | PART pruned (RWeka 패키지) |
| 74 | C5.0Rules | t | C5.0 규칙 집합 (C50 패키지) |
| 75 | JRip | t | RIPPER, numOpt 튜닝 (RWeka 패키지) |
| 76 | JRip | w | RIPPER, 2회 최적화 실행 (Weka) |
| 77 | OneR | t | 1-Rule 분류기 (RWeka 패키지) |
| 78 | OneR | w | OneR, 최소 6개 객체/버킷 (Weka) |
| 79 | DTNB | w | Decision Table / Naive Bayes 하이브리드 (Weka) |
| 80 | Ridor | w | Ripple-Down Rule Learner (Weka) |
| 81 | ZeroR | w | 항상 최빈 클래스 예측 (Weka, 기준선) |
| 82 | DecisionTable | w | Decision Table Majority Classifier, BestFirst 탐색 (Weka) |
| 83 | ConjunctiveRule | w | AND 연접 단일 규칙 (Weka) |

---

## 7. Boosting (BST) — 20개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 84 | adaboost | R | AdaBoost.M1, 분류 트리 기반 (adabag 패키지) |
| 85 | logitboost | R | LogitBoost, DecisionStump 기반, 200회 반복 (caTools 패키지) |
| 86 | LogitBoost | w | Additive Logistic Regression, DecisionStump 기반 (Weka) |
| 87 | RacedIncrementalLogitBoost | w | 점진적 학습 Raced LogitBoost (Weka) |
| 88 | AdaBoostM1 DecisionStump | w | AdaBoost.M1 + DecisionStump (Weka) |
| 89 | AdaBoostM1 J48 | w | AdaBoost.M1 + J48 (Weka) |
| 90 | C5.0 | t | C5.0 Boosting 앙상블, trials 튜닝 (C50 패키지) |
| 91 | MultiBoostAB DecisionStump | w | MultiBoost + DecisionStump (Weka) |
| 92 | MultiBoostAB DecisionTable | w | MultiBoost + DecisionTable (Weka) |
| 93 | MultiBoostAB IBk | w | MultiBoost + IBk (Weka) |
| 94 | MultiBoostAB J48 | w | MultiBoost + J48 (Weka) |
| 95 | MultiBoostAB LibSVM | w | MultiBoost + LibSVM (Weka) |
| 96 | MultiBoostAB Logistic | w | MultiBoost + Logistic (Weka) |
| 97 | MultiBoostAB MultilayerPerceptron | w | MultiBoost + MLP (Weka) |
| 98 | MultiBoostAB NaiveBayes | w | MultiBoost + NaiveBayes (Weka) |
| 99 | MultiBoostAB OneR | w | MultiBoost + OneR (Weka) |
| 100 | MultiBoostAB PART | w | MultiBoost + PART (Weka) |
| 101 | MultiBoostAB RandomForest | w | MultiBoost + RandomForest (Weka) |
| 102 | MultiBoostAB RandomTree | w | MultiBoost + RandomTree (Weka) |
| 103 | MultiBoostAB REPTree | w | MultiBoost + REPTree (Weka) |

---

## 8. Bagging (BAG) — 24개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 104 | bagging | R | Bagging 결정 트리 앙상블 (ipred 패키지) |
| 105 | treebag | t | Bagging 분류 트리 (ipred 패키지, caret) |
| 106 | ldaBag | R | Bagging of LDA (caret bag 함수) |
| 107 | plsBag | R | Bagging of PLS (caret bag 함수) |
| 108 | nbBag | R | Bagging of Naive Bayes (caret bag 함수) |
| 109 | ctreeBag | R | Bagging of Conditional Inference Tree (caret bag 함수) |
| 110 | svmBag | R | Bagging of SVM (caret bag 함수) |
| 111 | nnetBag | R | Bagging of MLP (caret bag 함수) |
| 112 | MetaCost | w | Cost-sensitive Bagging, ZeroR 기반 (Weka) |
| 113 | Bagging DecisionStump | w | Bagging + DecisionStump (Weka) |
| 114 | Bagging DecisionTable | w | Bagging + DecisionTable (Weka) |
| 115 | Bagging HyperPipes | w | Bagging + HyperPipes (Weka) |
| 116 | Bagging IBk | w | Bagging + IBk KNN (Weka) |
| 117 | Bagging J48 | w | Bagging + J48 (Weka) |
| 118 | Bagging LibSVM | w | Bagging + LibSVM Gaussian 커널 (Weka) |
| 119 | Bagging Logistic | w | Bagging + Logistic Regression (Weka) |
| 120 | Bagging LWL | w | Bagging + LocallyWeightedLearning (Weka) |
| 121 | Bagging MultilayerPerceptron | w | Bagging + MLP (Weka) |
| 122 | Bagging NaiveBayes | w | Bagging + NaiveBayes (Weka) |
| 123 | Bagging OneR | w | Bagging + OneR (Weka) |
| 124 | Bagging PART | w | Bagging + PART (Weka) |
| 125 | Bagging RandomForest | w | Bagging + RandomForest 500트리 (Weka) |
| 126 | Bagging RandomTree | w | Bagging + RandomTree (Weka) |
| 127 | Bagging REPTree | w | Bagging + REPTree (Weka) |

---

## 9. Stacking (STC) — 2개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 128 | Stacking | w | Stacking 앙상블, ZeroR 메타·기본 분류기 (Weka) |
| 129 | StackingC | w | 효율적 Stacking, 선형 회귀 메타 분류기 (Weka) |

---

## 10. Random Forests (RF) — 8개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 130 | rforest | R | Random Forest, ntree=500, mtry=√#inputs (randomForest 패키지) |
| 131 | rf | t | Random Forest, mtry 튜닝 (randomForest + caret) |
| 132 | RRF | t | Regularized Random Forest, coefReg·coefImp 튜닝 (RRF 패키지) |
| 133 | cforest | t | Conditional Inference Forest, mtry 튜닝 (party 패키지) |
| 134 | parRF | t | Parallel Random Forest, mtry 튜닝 (randomForest + caret) ⭐ **전체 1위** |
| 135 | RRFglobal | t | Global Regularized RF, coefReg 튜닝 (RRF 패키지) |
| 136 | RandomForest | w | RandomTree 기반 500트리 포레스트 (Weka) |
| 137 | RotationForest | w | Rotation Forest, J48 + PCA 필터 (Weka) |

---

## 11. Other Ensembles (OEN) — 11개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 138 | RandomCommittee | w | RandomTree 앙상블, 출력 평균 (Weka) |
| 139 | OrdinalClassClassifier | w | 순서형 분류 앙상블, J48 기반 (Weka) |
| 140 | MultiScheme | w | 교차 검증으로 ZeroR 분류기 선택 (Weka) |
| 141 | MultiClassClassifier | w | One-Against-All Logistic 이진 분류기 (Weka) |
| 142 | CostSensitiveClassifier | w | 비용 가중 ZeroR 앙상블 (Weka) |
| 143 | Grading | w | Grading 앙상블, ZeroR 기반 (Weka) |
| 144 | END | w | Ensemble of Nested Dichotomies, J48 이진 분류기 (Weka) |
| 145 | Decorate | w | 15개 J48 다양성 앙상블, 인공 훈련 패턴 활용 (Weka) |
| 146 | Vote | w | ZeroR 앙상블, 평균 규칙 결합 (Weka) |
| 147 | Dagging | w | SMO 앙상블, 4 fold 분할 (Weka) |
| 148 | LWL | w | Local Weighted Learning, DecisionStump 기반 (Weka) |

---

## 12. Generalized Linear Models (GLM) — 5개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 149 | glm | R | Generalized Linear Model (stats 패키지) |
| 150 | glmnet | R | GLM + Lasso/Elastic-net 정규화 (glmnet 패키지) |
| 151 | mlm | R | Multi-Log Linear Model, MLP 기반 (nnet 패키지) |
| 152 | bayesglm | t | Bayesian GLM, 기대값 최대화 방법 (arm 패키지) |
| 153 | glmStepAIC | t | AIC 기반 모델 선택 GLM (MASS 패키지) |

---

## 13. Nearest Neighbors (NN) — 5개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 154 | knn | R | K-Nearest Neighbors, k 튜닝 (class 패키지) |
| 155 | knn | t | KNN, k 튜닝 (caret 패키지) |
| 156 | NNge | w | Non-Nested Generalized Exemplars NN (Weka) |
| 157 | IBk | w | KNN, 교차 검증으로 K 튜닝, 유클리드 거리 (Weka) |
| 158 | IB1 | w | 1-NN 분류기 (Weka) |

---

## 14. Partial Least Squares & Principal Component Regression (PLSR) — 6개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 159 | pls | t | PLS Regression, 컴포넌트 수 튜닝 (pls 패키지) |
| 160 | gpls | R | Generalized PLS (gpls 패키지) |
| 161 | spls | R | Sparse PLS, K·eta 파라미터 튜닝 (spls 패키지) |
| 162 | simpls | R | SIMPLS 방법 PLS Regression (pls 패키지) |
| 163 | kernelpls | R | 커널 PLS, 최대 8개 주성분 (pls 패키지) |
| 164 | widekernelpls | R | Wide Kernel PLS, #inputs > #patterns 상황에 적합 (pls 패키지) |

---

## 15. Logistic & Multinomial Regression (LMR) — 3개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 165 | SimpleLogistic | w | 선형 로지스틱 회귀, LogitBoost 기반 (Weka) |
| 166 | Logistic | w | Multinomial Logistic Regression, ridge 추정기 (Weka) |
| 167 | multinom | t | Multinomial Log-Linear Model, MLP 기반, decay 튜닝 (nnet 패키지) |

---

## 16. Multivariate Adaptive Regression Splines (MARS) — 2개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 168 | mars | R | MARS 모델 (mda 패키지) |
| 169 | gcvEarth | t | Fast MARS, 교호작용 없는 덧셈 모델 (earth 패키지) |

---

## 17. Other Methods (OM) — 10개

| # | 이름 | 구현 | 설명 |
|---|------|------|------|
| 170 | pam | t | Nearest Shrunken Centroids (pamr 패키지) |
| 171 | VFI | w | Voting Feature Intervals (Weka) |
| 172 | HyperPipes | w | 클래스별 입력 범위 기반 분류 (Weka) |
| 173 | FilteredClassifier | w | Discretize 필터 후 J48 트리 (Weka) |
| 174 | CVParameterSelection | w | 10-fold CV로 ZeroR 파라미터 선택 (Weka) |
| 175 | ClassificationViaClustering | w | SimpleKMeans 클러스터링 기반 분류 (Weka) |
| 176 | AttributeSelectedClassifier | w | CfsSubsetEval 속성 선택 후 J48 (Weka) |
| 177 | ClassificationViaRegression | w | 클래스 이진화 후 M5P 회귀 모델 (Weka) |
| 178 | KStar | w | 엔트로피 기반 유사도 인스턴스 분류기 (Weka) |
| 179 | gaussprRadial | t | Gaussian Process 분류기, RBF 커널, sigma 튜닝 (kernlab 패키지) |

---

## 요약: 가족별 분류기 수

| 가족 | 약어 | 개수 |
|------|------|------|
| Discriminant Analysis | DA | 20 |
| Bayesian | BY | 6 |
| Neural Networks | NNET | 21 |
| Support Vector Machines | SVM | 10 |
| Decision Trees | DT | 14 |
| Rule-Based | RL | 12 |
| Boosting | BST | 20 |
| Bagging | BAG | 24 |
| Stacking | STC | 2 |
| Random Forests | RF | 8 |
| Other Ensembles | OEN | 11 |
| Generalized Linear Models | GLM | 5 |
| Nearest Neighbors | NN | 5 |
| Partial Least Squares & PCR | PLSR | 6 |
| Logistic & Multinomial Regression | LMR | 3 |
| MARS | MARS | 2 |
| Other Methods | OM | 10 |
| **합계** | | **179** |
