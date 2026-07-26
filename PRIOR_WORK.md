# 선행 연구 및 상용 시스템 대비 위치

> 조사 기준일: 2026-07-26. 모든 수치에 출처를 명시했다. 확인하지 못한 항목은 `⚠️ 미확인`으로 표시했고 추정치는 쓰지 않았다.
>
> 조사 방법: 로컬 PDF 5편(`/Users/hanbyulkang/Desktop/T1D/research papers/`), CEUR-WS Vol-2675(BGLP Challenge 2020 전문 17편 직접 다운로드·본문 추출), PubMed E-utilities(초록 원문). WebSearch는 세션 예산 소진으로 사용 불가였고, 대신 URL 직접 조회로 대체했다.

---

## 1. 요약: 우리는 어디쯤인가

| 항목 | 우리 결과 | 문헌 대표 범위 | 판정 |
|---|---|---|---|
| 30분 RMSE (모델) | **18.85** mg/dL (TCN, CGM만, cross-patient, OpenAPS) | OhioT1DM 개인화: 17.45–21.7<br>메타분석 평균: 21.40 (SD 12.56) | 문헌 범위 **안쪽**, 중상위 |
| 30분 RMSE (persistence 기준선) | **23.16** mg/dL | OhioT1DM `t0`: 21.67–22.60 | 문헌과 **거의 동일** → 과제 난이도가 비슷하다는 방증 |
| persistence 대비 개선폭 | **-4.31** mg/dL (-18.6%) | Mirshekarian 2019: 22.60→19.07 (-15.6%)<br>Toledo-Marín 2023(OpenAPS): 개선 없음 | 문헌 **상단** |
| cross-patient 30분 RMSE | **18.85** | LOPO-CV(Ohio): 24.89 / 29.00 / 30.86<br>patient-excluded(Ohio): 18.32 | 상위권이나 **데이터셋이 달라 직접 비교 불가** |
| 저혈당 재현율 @ 거짓경보 | 74.8% (예산 6/일, **실측 14.7/일**)<br>59.9% (예산 3/일, **실측 8.1/일**) | sens 85–90% @ FPR 21–38% (비율 기준) | **직접 비교 불가**, 문헌은 FA를 "비율"로, 우리는 "1일 횟수"로 보고 |
| Clarke A+B | **96.61%** | 92%(LOPO-CV) / 97.3–99.3%(개인화) / 99.6% | 문헌 범위 안이나 **하단** |

**판단 3줄**

1. RMSE는 문헌 범위 안에 확실히 들어간다. 30분 horizon에서 CGM만 쓰는 딥러닝의 사실상 천장은 OhioT1DM 기준 **18–19 mg/dL 부근**이고(Mirshekarian 2019의 CGM-only LSTM 19.07, BGLP 2020 상위권 18.2–18.6), 우리 18.85는 그 대역 안이다. 단 **우리는 cross-patient, 저들은 개인화**라는 점이 우리에게 유리하게 작용하고, **OpenAPS는 폐쇄루프 사용자 데이터라 혈당이 안정적**이라는 점이 우리에게 불리하게(=쉽게) 작용한다. 두 효과의 크기를 분리하지 못했으므로 "SOTA를 이겼다"는 주장은 성립하지 않는다.
2. 같은 데이터셋(OpenAPS Data Commons)을 쓴 유일한 공개 비교 대상인 Toledo-Marín et al. 2023은 30분 RMSE 24 mg/dL이었고, **딥러닝이 last-measurement(persistence) 대비 유의한 개선을 내지 못했다**고 결론지었다. 우리는 같은 데이터셋에서 persistence를 18.6% 개선했다. 이것이 우리 결과의 가장 방어 가능한 주장이다.
3. 저혈당 경보 쪽은 **문헌 자체가 비교 가능한 형태로 보고하지 않는다.** 30분 horizon에서 sensitivity와 false alarm rate를 함께 보고한 연구는 소수이고, "하루 몇 번 울리는가"를 보고한 연구는 조사 범위에서 사실상 1편뿐이었다(Prendin 2025, 그것도 T1D가 아님). 우리가 FA/day를 명시한 것은 문헌 대비 **더 엄격한** 보고이며, 동시에 **직접 비교 대상이 없다**는 뜻이다.

---

## 2. 30분 예측 RMSE 벤치마크

### 2.1 OhioT1DM: 기준선 및 개인화 모델

| 연구 | 데이터셋 | 환자수 | 분할 방식 | 입력 | 30분 RMSE (mg/dL) | 출처 |
|---|---|---|---|---|---|---|
| Mirshekarian et al. 2019, `t0` (persistence) | OhioT1DM 2018 | 6 | 환자별(개인화) | CGM | **22.60** (agnostic) / 21.67 (inertial) | Mirshekarian, Shen, Bunescu, Marling, *EMBC 2019*, Table IV |
| 〃 ARIMA | 〃 | 6 | 개인화 | CGM | 20.17 / 19.36 | 〃 |
| 〃 LSTM (dropout 0.1) | 〃 | 6 | 개인화 | **CGM만** | **19.07** / 18.72 | 〃 |
| 〃 LSTM | 〃 | 6 | 개인화 | CGM+인슐린+식사 | 18.74 / 18.07 | 〃 Table V |
| 〃 double LSTM (What-if) | 〃 | 6 | 개인화 | CGM+I+M(+생체신호) | **18.10–18.19** | 〃 Table VI–VII |
| Nemat et al. 2024, ARIMA / SVR / LSTM 평균 | OhioT1DM 2018 | 6 | 개인화(환자별 하이퍼파라미터 튜닝) | CGM(univariate) | 20.39–20.78 | Nemat, Khadem, Elliott, Benaissa, *Sci Rep* 14:21863 (2024), Table 4 |
| 〃 (multivariate) | 〃 | 6 | 개인화 | CGM+Carb+Bolus+활동 | 19.59–21.11 | 〃 |
| Nemat et al. 2024 | OhioT1DM 2020 | 6 | 개인화 | CGM | 20.03–20.83 | 〃 Table 6 |

### 2.2 BGLP Challenge

**BGLP Challenge 2018** (IJCAI 2018, 7개 팀, OhioT1DM 2018 6명)

| 항목 | 값 | 출처 |
|---|---|---|
| 참가 팀 수 | 7팀 | Mirshekarian et al. 2019, §C |
| Mirshekarian 팀 (원 마감일 기준 1위) 30분 RMSE | **18.11–18.19** | 〃 ("rank us first among results submitted by the original deadline") |
| 2018 챌린지 결과의 30분 RMSE 범위 | **18.9 – 21.7** | Rubin-Falcone, Fox, Wiens, *KDH/BGLP 2020*, CEUR-WS Vol-2675 pp.85-89 본문: "comparable to the results achieved in the 2018 challenge (i.e. 18.9 to 21.7 for 30 minute rMSE)" |

> 주: Mirshekarian 저자들 스스로 "참가 시스템들이 반드시 동일한 실험 설정으로 평가된 것은 아니므로 순위는 신중히 봐야 한다"고 명시했다.

**BGLP Challenge 2020** (KDH@ECAI 2020, OhioT1DM 2020 cohort 6명 test). 아래는 CEUR-WS Vol-2675 전문에서 직접 추출한 값이다.

| 팀/논문 | 방법 | 분할 방식 | 30분 RMSE (mg/dL) | 출처(CEUR-WS Vol-2675) |
|---|---|---|---|---|
| Freiburghaus, Rizzotti-Kaddouri, Albertetti | Deep learning | 개인화 | **17.45** | paper23, pp.116-120 |
| Rubin-Falcone, Fox, Wiens (Michigan) | Deep residual (N-BEATS 계열) + 사전학습 | 개인화 | **18.2** (baseline 21.2) | paper18, pp.96-100 |
| **Bevan & Coenen, all patients** | LSTM(128) | **전 환자 통합** | **18.23** (±2.36) | paper17, pp.91-95, Table 3 |
| **Bevan & Coenen, patient excluded** | LSTM(128) | **cross-patient (해당 환자 데이터 제외)** | **18.32** (±2.40) | 〃 |
| **Bevan & Coenen, patient only** | LSTM(128) | 개인화 | 19.21 (±2.48) | 〃 |
| Zhu, Yao, Li, Herrero, Georgiou | GAN | 개인화 | 18.34 ± 0.17 | paper15, pp.86-90 |
| Pavan et al. (NN-EIM) | Shallow NN + error imputation | 개인화 | 18.63 (CGM-NN 19.50 대비) | paper16, pp.91-95 |
| Nemat et al. | 활동+CGM 데이터 융합 | 개인화 | 18.99 | paper21 |
| Khadem, Nemat et al. | Multi-lag stacking | 개인화 | 19.01 | paper26 |
| Yang et al. | MS-LSTM (multi-lag) | 개인화 | 19.048 | paper24 |
| Hameed & Kleinberg | Knowledge distillation | 개인화 | 19.21 (**CGM 단독 입력**) | paper14 |
| Daniels, Herrero, Georgiou | Deep multitask | 개인화 | 19.79 ± 0.06 | paper19 |
| Cappon et al. | 개인화 해석가능 DL | 개인화 | 20.20 | paper12 |
| Sun et al. | Latent-variable model | 개인화 | 16.66 – 22.76 (환자별 범위) | paper20 |
| Xie et al. (ARMA+보상망) | ARMA + residual NN | 개인화 | 21.44 – 21.80 | paper27 |

> **BGLP 2020의 공식 순위/우승팀 발표는 확인하지 못했다 (⚠️ 미확인).** 위 표는 개별 논문이 스스로 보고한 값이며 전처리·결측 처리·평가 구간이 팀마다 달라 엄밀한 순위표가 아니다.

### 2.3 서베이/메타분석이 보고하는 범위

| 출처 | 대상 | 30분 horizon RMSE |
|---|---|---|
| Liu K, Li L, Ma Y, et al. *JMIR Med Inform* 2023;11:e47833 (Systematic Review & Network Meta-Analysis) | 당뇨 혈당 예측 ML 모델 전반 | **평균 21.40 mg/dL (SD 12.56)**<br>(15분 18.88 SD 19.71 / 45분 21.27 SD 5.17 / 60분 30.01 SD 7.23) |
| Kapoor Y, Hasija Y. *Technol Health Care* 2025;33(1):577-591 (meta-analysis, 10편) | CGM + IoT ML 모델 | **평균 21.488 mg/dL (SD 2.92)**<br>(15분 15.02 / 45분 30.09 / 60분 35.89) |

### 2.4 cross-patient(비개인화) 평가를 명시한 연구

| 연구 | 데이터셋 | 환자수 | 분할 | 30분 RMSE | 출처 |
|---|---|---|---|---|---|
| Moon, Kim, Yoo, Cho 2025. BiT-MAML | OhioT1DM 2018 | 6 | **LOPO-CV (leave-one-patient-out)** | **24.89 ± 4.60** (환자별 19.64–30.57) | *Sci Rep* 15:30636 (2025), Table 5–6 |
| 〃 Edge-LSTM 재구현 baseline | 〃 | 6 | LOPO-CV | 29.00 ± 7.04 | 〃 |
| 〃 LSTM 재구현 baseline | 〃 | 6 | LOPO-CV | 30.86 ± 5.78 | 〃 |
| Bevan & Coenen 2020 (patient excluded) | OhioT1DM 2020 | 6 | cross-patient | 18.32 | CEUR-WS Vol-2675 paper17 |
| Toledo-Marín, Ali, van Rooij, Görges, Wasserman 2023, CNN | **OpenAPS Data Commons** | **139** | train 7% / test 93% (환자 단위 여부 ⚠️ 미확인) | **24** (15분 16 / 60분 37). **딥러닝이 last-measurement 대비 유의한 개선 없음**이라고 결론 | *J Clin Med* 2023;12(4):1695 |
| **우리 (TCN)** | **OpenAPS Data Commons** | **40 (test 8)** | **환자 단위 cross-patient** | **18.85** (persistence 23.16) | `results.md` |

### 2.5 Clarke Error Grid A+B 벤치마크

| 연구 | 조건 | A+B | 출처 |
|---|---|---|---|
| Moon et al. 2025 (BiT-MAML) | Ohio, PH 30, **LOPO-CV**, 35,000+ 예측점 | **>92%** (D+E <0.2%) | *Sci Rep* 15:30636 |
| Mirshekarian et al. 2019 (LSTM) | Ohio, What-If, **PH 60**, 개인화 | **99.26%** (A 12,627 / B 2,733 / D 115) | *EMBC 2019* Table VIII (수치로부터 계산) |
| 〃 ARIMA 기준선 | 〃 | 97.31% (A 11,236 / B 3,823 / C 60 / D 353 / E 3) | 〃 |
| Xiong et al. 2025 (Ls-Encoder) | 자체 임상 데이터셋 13명, PH 30 | 99.60% (PH 120에서도 96.37%) | *DIGITAL HEALTH*, Table 4 |
| **우리 (TCN)** | **OpenAPS, PH 30, cross-patient** | **96.61%** (persistence 95.14%) | `results.md` |

> **A+B에 대한 공식 규제 기준은 "예측 모델"에 대해서는 존재하지 않는다.** Clarke EGA는 원래 혈당 자가측정기(SMBG) 평가용으로 제안되었고(Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL. *Diabetes Care* 1987;10(5):622-8), ISO 15197 계열 기준은 오차 그리드가 아니라 절대/상대 오차 한계로 규정된다(예: DIN EN ISO 15197:2003, 결과의 ≥95%가 <75 mg/dL 구간에서 ±15 mg/dL, ≥75 mg/dL 구간에서 ±20% 이내. Freckmann G et al. *Diabetes Technol Ther* 2010;12(3):221-31). 따라서 "A+B 몇 % 이상이면 합격"이라는 인용 가능한 임계값은 없다. 실무적으로 이 분야 논문들은 92–99.6% 범위를 보고한다.

---

## 3. 저혈당 예측 sensitivity / false alarm

sensitivity와 false alarm을 **함께** 보고한 연구만 정리했다.

| 연구 | 데이터/대상 | 예측 horizon | 임계값 | Sensitivity | False alarm 지표 | 출처 |
|---|---|---|---|---|---|---|
| Palerm CC, Bequette BW 2007 | hypoglycemic clamp 데이터 | **30분** | 70 mg/dL | **90%** | specificity 79% → 저자 표현 "**21% false alarm rate를 감수해야 한다**" | *J Diabetes Sci Technol* 2007;1(5):624-9 |
| Seo W, Lee YB, Lee S, Jin SM, Park SM 2019 | 식후 저혈당, T1D | **30분** | hypoglycemia alert value | **89.6%** (평균) | specificity 91.3%, **F1 0.543** (→ precision이 낮음을 시사) | *BMC Med Inform Decis Mak* 2019;19(1):210 |
| Li J, Ma X, et al. 2020 | 야간 저혈당 | **30분** |, | **>90.07%** | specificity **>87.79%** (15분 시점: sens >96.03%, spec >96.07%) | *J Diabetes Res* 2020:8830774 |
| Yu X, Ma N, Yang T, et al. 2021 | 임상 데이터, 다단계 조기경보 | 평균 조기경보 **20.61분** (level-I) | level-I | **85.90%** | **false-positive 23.86%**, miss 14.10% | *BMC Med Inform Decis Mak* 2021;21(1):22 |
| 〃 level-II | 〃 | 평균 27.66분 | level-II | 80.36% | false-positive 17.37% | 〃 |
| Fleischer J, Hansen TK, Cichosz SL 2022 | 225명, CGM 370만 포인트, 앙상블 | **40분** |, | **90%** (event 기준) | **false-positive rate 38%**, lead-time 17.5분, ROC AUC 0.988, PR AUC 0.767 | *Front Clin Diabetes Healthc* 2022;3:1066744 |
| Bertachi et al. (Xiong et al. 2025가 인용) | OhioT1DM, ANN+생리모델 | 야간 저혈당 검출 |, | **85%** | specificity **92%** | Xiong et al. *DIGITAL HEALTH* 2025, 서론부 인용. **원논문 직접 확인 못함 ⚠️** |
| Cameron F, Niemeyer G, Gundy-Burlet K, Buckingham B 2008 | DirecNet 입원 26건, Navigator 1분 데이터 | 0–20분 |, | 놓친 이벤트 **0건** | **PPV 60%로 설정**(경보의 60%가 실제 이벤트), 평균 lead time 23분, 오경보 시 최저혈당 평균 97 mg/dL | *J Diabetes Sci Technol* 2008;2(4):612-21 |
| Prendin F, et al. 2025 | **post-bariatric 저혈당(T1D 아님)**, 50명 | run-to-run rAR |, | recall **84.43%** | precision 64.38%, F1 73.06%, **6일에 1회 거짓경보** | *BMC Med Inform Decis Mak* 2025;25(1):33 |
| **우리 (tcn_prob)** | **OpenAPS, 미학습 환자 8명** | **30분** | 70 mg/dL | **74.8%** | 검증셋 기준 예산 6회/일 → **테스트 실측 14.7회/일**, precision 38.6% | `alarm.md` |
| **우리 (tcn_prob, 중간 작동점)** | 〃 | 30분 | 70 mg/dL | **59.9%** | 예산 3회/일 → **실측 8.1회/일**, precision 47.8% | 〃 |
| **우리 (tcn_prob, 보수 작동점)** | 〃 | 30분 | 70 mg/dL | 38.3% | 예산 1회/일 → 실측 3.3회/일, precision 59.2% | 〃 |

**해석 주의, 이 표는 그대로 비교하면 안 된다.**

- 문헌의 "false alarm rate"는 대부분 **비율**(1−specificity, 또는 FP/(FP+TN))이다. 우리 지표는 **하루 몇 번 울리는가**다. 두 값은 저혈당 사건의 기저율(prevalence)과 평가 단위(샘플 단위 vs 이벤트 단위)에 따라 전혀 다른 숫자가 된다.
- 우리 표의 precision 38.6–59.2%는 Prendin 2025의 64.38%, Cameron 2008의 PPV 60%와 **같은 축**이며, 그 관점에서는 우리 보수 작동점(precision 59.2% @ recall 38.3%)이 문헌과 같은 대역에 있다. 다만 recall이 낮다.
- Palerm 2007(sens 90% @ FPR 21%)은 clamp 실험 데이터라 실생활 데이터보다 훨씬 유리한 조건이다. 실생활 데이터인 Fleischer 2022는 같은 sens 90%에서 FPR이 38%로 뛴다. **실생활 조건에서 sens 90%는 대략 FPR 40% 수준의 대가를 요구한다**는 것이 현재 문헌의 실질적 상한선이다.

### 3.1 false alarm rate를 보고하지 않는 연구들

**이 분야의 지배적 관행은 sensitivity(또는 AUC)만 보고하고 false alarm 비용을 보고하지 않는 것이다.** 이것 자체가 확인된 사실이다.

| 근거 | 내용 | 출처 |
|---|---|---|
| 저혈당 예측 알고리즘 체계적 문헌고찰 | T1D 저혈당 예측 모델 **19개**를 검토했으나 성능 요약이 "**정확도 70%~99%**"라는 한 줄뿐이며, false alarm rate·precision·경보 빈도에 대한 종합은 제시되지 않음 | Tsichlaki S, Koumakis L, Tsiknakis M. *JMIR Diabetes* 2022;7(3):e34699 |
| 저혈당 예측 모델링 리뷰 (79편) | 5개 저혈당 유형별로 모델을 분류했으나 경보 부담(alarm burden)을 공통 지표로 다루지 않음 | Zhang L, Yang L, Zhou Z. *Front Public Health* 2023;11:1044059 |
| CGM 경보 평가 방법론 비판 | CGM 성능 논문 **129편 중 약 25%만** 경보 평가를 포함. **예측(predictive) 경보에 대해서는 "문헌에서 찾은 결과가 더 적다"**고 명시 | Pleus S, Eichenlaub M, Waldenmaier D, Freckmann G. *J Diabetes Sci Technol* 2024;18(4):847-856 |
| 혈당 예측 딥러닝 논문 다수 | Mirshekarian 2019, Nemat 2024, Moon 2025, 그리고 BGLP 2020 챌린지 논문 17편 전체, **RMSE/MAE와 (일부) Clarke EGA만 보고. 저혈당 경보의 sensitivity/false alarm을 보고한 논문은 없음** | 본 조사에서 전문 직접 확인 |
| ML 모델 메타분석 | 저혈당 예측에 대해 **likelihood ratio로만** 종합: LR+ 8.3 (95% CI 5.7–12.0), LR− 0.31 (0.22–0.44). 경보 빈도 지표 없음 | Liu K et al. *JMIR Med Inform* 2023;11:e47833 |

→ **결론: 우리가 `alarm.md`에서 "동일 false-alarm 예산 하의 recall"을 보고한 것은 이 분야의 표준보다 엄격한 보고 방식이다. 대신 직접 대응하는 선행 수치가 거의 없다.**

---

## 4. 상용 시스템 수치와 비교 가능성

### 4.1 Dexcom G6 / G7: "Urgent Low Soon"

| 항목 | 값 | 출처 |
|---|---|---|
| 알고리즘 사양 | 센서 혈당이 **20분 이내에 ≤55 mg/dL**에 도달할 것으로 예측되면 경보 | Acciaroli G, Welsh JB, Akturk HK. *J Diabetes Sci Technol* 2022;16(3):677-682 (저자 2인 Dexcom 직원); Puhr S, Derdzinski M, Welsh JB, et al. *Diabetes Technol Ther* 2019;21(4):155-158 |
| G5→G6 전환 실사용 효과 (n=1,424) | <54, ≤55, <70, >250 mg/dL 체류시간이 모두 유의하게 감소. 저임계 경보 70 mg/dL 설정 사용자에서는 TIR도 개선 | Puhr et al. 2019 DTT |
| G5→G6 전환, 반등 고혈당 (n=24,518) | 반등 고혈당 사건 빈도/지속/중증도가 각각 **-7% / -8% / -13%** (모두 P<.001) | Acciaroli et al. 2022 JDST |
| Libre 2 → G7 전환 (n=29, 12주) | TBR <70 mg/dL **3.0% → 2.0%** (P=0.006), TBR<4% 달성률 55.2%→82.8%, CV 39.3%→37.2% | Preechasuk L, Avari P, Oliver N, Reddy M. *Diabetes Technol Ther* 2024;26(7):498-502 |
| **공개된 sensitivity / false alarm rate** | **없음** | ↓ 아래 참조 |

### 4.2 Medtronic SmartGuard / PLGS·PLGM 임상시험

| 시험 | 설계 | 저혈당 감소 | 출처 |
|---|---|---|---|
| PLGM RCT (MiniMed 640G "Suspend before low") | 6개월, 다기관 RCT, 소아·청소년 **n=154** (SAPT 74 vs PLGM 80) | SG <63 mg/dL 체류시간 **2.6% → 1.5%** (P<0.0001).<br>저혈당 사건 **227 → 139 events/patient-year** (P<0.001, **-39%**).<br>HbA1c 차이 없음 | Abraham MB, Nicholas JA, Smith GJ, et al. *Diabetes Care* 2018;41(2):303-310 |
| PILGRIM (in silico + 운동 feasibility) | FDA 인증 시뮬레이터 가상환자 100명 + 청소년 22명 | in silico 저혈각(<70) **-26.7%** (LGS는 -5.3%). 저혈당 지속시간 중앙값 58분 vs LGS 101분 (P<0.001). 운동 시험에서 임계 도달 환자의 **80%에서 저혈당 예방** | Danne T, Tsioli C, Kordonouri O, et al. *Diabetes Technol Ther* 2014;16(6):338-47 |
| 640G-SmartGuard 후향 관찰 | 소아 **n=21**, 평균 5.0±2.1개월 | 저혈당 빈도 **10.4±5.2% → 7.6±3.3%** (P=.044). 고혈당 증가 없음. 평균 정지시간 3.1±1.2 h/일 | Villafuerte Quispe B, et al. *Endocrinol Diabetes Nutr* 2017;64(4):198-203 |
| SMILE (640G SmartGuard PLGM, 고위험군) | 24주 RCT 프로토콜 논문 (1차 결과: SG ≤55 mg/dL 20분 초과 사건 수) | 프로토콜만 확인, 결과 수치 **⚠️ 미확인** | De Valk HW, et al. *Diabetes Technol Ther* 2018;20(11):758-766 |

### 4.3 Tandem Basal-IQ (참고: 동일 계열 PLGS)

| 시험 | 설계 | 결과 | 출처 |
|---|---|---|---|
| PROLOG | 6주 무작위 교차, **n=103** (6–72세), t:slim X2 + Dexcom G5 | <70 mg/dL 체류시간 **4.5% → 3.1% (평균 기준 -31%)**, 중앙값 3.2%→2.6% (P<0.001). 반등 고혈당 없음. 평균 펌프 정지 104분/일 | Forlenza GP, Li Z, Buckingham BA, et al. *Diabetes Care* 2018;41(10):2155-2161 |

### 4.4 Abbott FreeStyle Libre

| 항목 | 값 | 출처 |
|---|---|---|
| FreeStyle Libre 2 | **임계값 기반 선택적 경보만 제공. 예측 경보 없음** ("isCGM with glucose threshold-based optional alerts only") | Preechasuk et al. *Diabetes Technol Ther* 2024;26(7):498-502 |
| FreeStyle Libre 3 / 3 Plus 예측 경보 유무 및 사양 | **⚠️ 미확인** |, |

### 4.5 직접 비교 가능성: 결론

**불가능하다.** 근거는 방법론 리뷰 한 편에 명확히 정리되어 있다.

> Pleus S, Eichenlaub M, Waldenmaier D, Freckmann G. "A Critical Discussion of Alert Evaluations in the Context of Continuous Glucose Monitoring System Performance." *J Diabetes Sci Technol* 2024;18(4):847-856.
> - "retrospectively determining predictive CGM alerts can be difficult, as the specific way in which a CGM system triggers predictive alerts may be **proprietary information**."
> - 많은 시스템이 경보 발생을 다운로드 가능한 데이터에 **기록하지 않음** → 수동 기록 필요.
> - 제조사가 사용자에게 **단일 임계값만 허용** → 다중 작동점 평가 불가.
> - Dexcom G7의 "urgent low soon"에 대한 제조사 후원 연구도 **detection rate(sensitivity)나 false alarm 빈도를 보고하지 않음**.
> - Medtronic Guardian 4는 예측 horizon을 10–60분으로 조절 가능하지만 **공개된 성능 데이터가 전혀 없음**.

정리하면:

| 비교 축 | 우리 | 상용 시스템 | 비교 가능? |
|---|---|---|---|
| 예측 horizon | 30분 | Dexcom 20분 / Medtronic 조정 가능 | ✗ 다름 |
| 임계값 | <70 mg/dL | Dexcom ≤55 mg/dL (긴급 저혈당) | ✗ 다름 (55는 훨씬 드문 사건 → 경보 통계가 완전히 다름) |
| sensitivity / FA rate | 보고함 | **미공개** | ✗ 상대가 없음 |
| 임상 결과(TBR 감소) | 측정 불가(개입 없음) | Abraham -42%(2.6→1.5), Forlenza -31% | ✗ 우리는 개입 시험이 아님 |

**따라서 "우리 모델이 Dexcom보다 낫다/못하다"는 문장은 어떤 형태로도 쓸 수 없다.** 쓸 수 있는 문장은 "상용 예측 경보의 sensitivity/false alarm은 공개되지 않았으며(Pleus 2024), 우리는 그 지표를 공개한다" 정도다.

---

## 5. personalized vs cross-patient 평가 관행

### 5.1 이 분야의 표준은 personalized(개인화)다

- OhioT1DM 데이터셋 자체가 환자별 train/test 분할로 배포되며, BGLP Challenge 규칙도 환자별 예측이다. 본 조사에서 전문을 확인한 **BGLP 2020 논문 17편 중 16편이 개인화(환자별 모델 또는 환자별 하이퍼파라미터 튜닝)** 였고, 명시적으로 비개인화를 다룬 논문은 Bevan & Coenen 1편뿐이다(제목부터 "Experiments in **non-personalized** future blood glucose level prediction").
- Nemat et al. 2024는 ARIMA의 (p,d,q)와 SVR의 (γ, C, ε)를 **환자마다 개별 탐색**했다 (*Sci Rep* 14:21863, Table 2–3). 이것이 이 분야의 기본값이다.
- 이 관행의 대가는 Moon et al. 2025가 명시적으로 지적한다: 개인화 논문들이 보고하는 우수한 수치는 "미학습 환자로의 일반화"를 측정하지 않는다는 것.

### 5.2 cross-patient가 얼마나 불리한가: 증거가 엇갈린다

| 증거 | 방향 | 수치 |
|---|---|---|
| **Bevan & Coenen 2020** (CEUR-WS Vol-2675 paper17, OhioT1DM 2020, n=6) | **cross-patient가 오히려 유리** | patient-only **19.21** → patient-excluded **18.32** → all-patients **18.23**.<br>저자 결론: "해당 환자 데이터를 제외하고 대량 데이터로 학습한 모델이 **그 환자 자신의 데이터만으로 학습한 모델을 유의하게 능가**했다(p=0.05)". 또 all-patients와 patient-excluded 사이에는 **유의차 없음**. |
| **Moon et al. 2025** (*Sci Rep* 15:30636, OhioT1DM 2018, n=6, LOPO-CV) | **cross-patient가 크게 불리** | LOPO-CV에서 LSTM **30.86**, Edge-LSTM **29.00**, 제안 모델(메타러닝) **24.89**.<br>동일 데이터셋의 개인화 문헌값(Edge-LSTM 19.10 ± 2.04, Martinsson LSTM 18.86 ± 1.79. Moon et al. Table 1)과 비교하면 **약 +6 ~ +12 mg/dL 열화**. |
| **Toledo-Marín et al. 2023** (OpenAPS, n=139) | 대규모 데이터에서는 딥러닝의 이점 자체가 사라질 수 있음 | CNN 30분 RMSE 24, 그러나 **last-measurement 대비 유의한 개선 없음** |

**해석:** 두 결과의 차이는 학습 데이터 규모와 모델 용량으로 설명하는 것이 자연스럽다. Bevan은 단일 LSTM(128 유닛)에 12명 전체 데이터를 통합해 학습했고, Moon은 6명 중 5명으로 학습해 1명에 적용했다(LOPO). 즉 **cross-patient 자체가 본질적으로 불리한 것이 아니라, 학습 환자 수가 적을 때 불리하다**는 쪽이 증거에 부합한다.

**우리 설정에 대한 함의:** 우리는 학습 26명 / 검증 6명 / 테스트 8명, 총 28,281 환자-일이다. OhioT1DM(12명 × 8주 ≈ 672 환자-일)보다 **40배 이상 크다**. Bevan의 결과에 비추면 이 규모에서 cross-patient가 개인화 대비 심각한 불리함을 안는다고 볼 근거는 약하다. 다만 `alarm.md`에서 관찰된 **경보 임계값의 검증→테스트 전이 실패**(예산 3회/일 → 실측 8.1회/일)는 cross-patient 설정의 실제 비용이 **RMSE가 아니라 캘리브레이션에서 나타난다**는 것을 보여준다. 이는 Moon et al.이 보고한 환자 간 편차(BiT-MAML 환자별 RMSE 19.64–30.57, Clarke Zone A 71.9%–92.0%)와 같은 현상이다.

---

## 6. 우리 결과의 정직한 위치

### 유리한 점

1. **기준선이 문헌과 일치한다.** 우리 persistence 30분 RMSE 23.16은 Mirshekarian et al.의 `t0` 21.67–22.60과 거의 같다. 과제 난이도가 극단적으로 쉽거나 어렵지 않다는 최소한의 sanity check가 성립한다.
2. **같은 데이터셋의 유일한 공개 선행연구를 명확히 앞선다.** Toledo-Marín et al. 2023은 OpenAPS Data Commons(139명)에서 CNN 30분 RMSE 24 mg/dL을 얻었고 **persistence 대비 유의한 개선을 내지 못했다**고 결론지었다. 우리는 18.85로 persistence(23.16)를 18.6% 개선했다. 이것이 가장 방어 가능한 주장이다.
3. **평가 프로토콜이 문헌 표준보다 엄격하다.** (a) 환자 단위 분할, (b) 검증 환자와 테스트 환자 완전 분리, (c) 선택은 검증에서만, (d) CGM 단독 입력. BGLP 2020 논문 17편 중 16편이 개인화였다는 점과 대비된다.
4. **데이터 규모가 크다.** 40명 / 28,281 환자-일. OhioT1DM(12명 / 약 672 환자-일)의 40배 이상. Bevan & Coenen이 보인 "데이터가 많으면 비개인화가 개인화를 이긴다"는 결과와 정합적이다.
5. **경보 지표를 이 분야가 보고하지 않는 형태로 보고했다.** 체계적 문헌고찰(Tsichlaki 2022, 19개 모델)조차 "정확도 70–99%"로만 요약하고, CGM 경보 평가 리뷰(Pleus 2024)는 예측 경보의 정량 평가가 거의 없다고 명시한다. 우리는 동일 false-alarm 예산 하에서 모델을 비교했다.
6. **RMSE와 저혈당 재현율의 역상관을 명시적으로 보고했다.** 이 트레이드오프를 표로 드러낸 선행 연구를 조사 범위에서 찾지 못했다.

### 불리한 점

1. **RMSE 절대값의 dataset-대-dataset 비교가 성립하지 않는다.** OpenAPS Data Commons는 **오픈소스 폐쇄루프(AID) 사용자**들이 기증한 데이터다(Shahid A, Lewis DM. *Nutrients* 2022;14(9):1906, "open-source AID users"). 이들은 TIR/TBR/TAR이 권장 목표 안에 있는 집단으로 보고되었다(Cooper D, Reinhold B, Shahid A, Lewis DM. *J Diabetes Sci Technol* 2025;19(3):649-657). OhioT1DM 참가자는 Medtronic 530G/630G(비폐쇄루프 SAP) 사용자다(Marling & Bunescu, CEUR-WS Vol-2675 pp.71-74). **혈당 변동성이 낮은 집단은 예측이 쉽다.** 우리 persistence 23.16이 Ohio의 22.60과 비슷하다는 점이 이 우려를 부분적으로 완화하지만 해소하지는 않는다.
2. **OhioT1DM에서 평가하지 않았다.** 이 분야의 사실상 공통 벤치마크에서 수치가 없으므로 "SOTA와 비교"라는 문장을 쓸 수 없다.
3. **테스트 환자가 8명뿐이다.** Moon et al. 2025는 n=6 LOPO-CV에서 제안 모델이 baseline을 이겼음에도 **통계적 유의성에 도달하지 못했다**(paired t-test p=0.19, p=0.29)고 정직하게 보고했다. 우리 n=8도 같은 한계에 노출된다. 환자별 RMSE 분산과 유의성 검정을 보고하지 않으면 같은 비판을 받는다.
4. **Clarke A+B 96.61%는 문헌 범위의 하단이다.** 개인화 연구들은 97.3–99.6%를 보고한다. 우리보다 나은 유일한 cross-patient 비교 대상인 Moon et al.의 >92%보다는 높지만, "임상적으로 안전"이라는 주장을 하기에는 근거가 약하다.
5. **경보 임계값이 검증→테스트로 전이되지 않는다.** 예산 1/3/6회/일로 맞춘 임계값이 테스트에서 3.3/8.1/14.7회/일로 나타났다. 즉 요약에서 "6회/일에 재현율 74.8%"라고 쓰면 **부정확하다**, 정확한 서술은 "검증셋에서 6회/일이 되도록 맞춘 임계값을 테스트에 적용했더니 재현율 74.8%, 실제 거짓경보 14.7회/일"이다. 하루 14.7회 경보는 착용 가능한 제품이 아니다.
6. **실생활 저혈당 예측의 문헌 상한선과 비교하면 우수하지 않다.** Fleischer et al. 2022는 실생활 데이터 225명에서 40분 horizon, sensitivity 90% @ FPR 38%를 얻었다. 우리는 30분 horizon에서 74.8%가 상한이다. horizon과 지표 정의가 달라 직접 비교는 불가하지만, "문헌을 앞선다"는 주장의 근거는 없다.
7. **상용 시스템과의 비교가 원천적으로 불가능하다.** §4.5 참조.

### 정직한 한 문장

> CGM만으로 30분 후 혈당을 예측하는 과제에서, 우리 결과(cross-patient RMSE 18.85, persistence 23.16, Clarke A+B 96.6%)는 OhioT1DM 기반 문헌의 개인화 모델 성능 대역(17.5–21.7) 안에 있고, 같은 OpenAPS 데이터셋의 유일한 선행연구(RMSE 24, persistence 대비 개선 없음)를 명확히 앞선다. 다만 데이터셋이 다르고(폐쇄루프 사용자 집단), 테스트 환자가 8명이며, 저혈당 경보의 임계값이 집단 수준에서 전이되지 않는다는 점에서 "SOTA" 주장은 성립하지 않는다.

---

## 7. ⚠️ 미확인 항목

| # | 항목 | 왜 미확인인가 | 대체로 사용한 근거 |
|---|---|---|---|
| 1 | **BGLP Challenge 2018 공식 순위표/결과 요약 원문** | 챌린지 결과 요약 논문을 찾지 못함 | Mirshekarian et al. 2019 본문의 "7개 팀 중 1위, RMSE 18.11–18.19" 진술 + Rubin-Falcone et al. 2020의 "2018 챌린지 30분 rMSE 18.9–21.7" 인용 |
| 2 | **BGLP Challenge 2020 공식 순위/우승팀** | CEUR-WS Vol-2675에 결과 요약 논문 없음 (Marling & Bunescu의 dataset update 논문만 존재) | 개별 논문 17편의 자체 보고 수치를 직접 추출해 표로 제시 |
| 3 | **Dexcom G6/G7 "Urgent Low Soon"의 sensitivity / false alarm rate** | 제조사가 공개하지 않음 | Pleus et al. 2024 JDST가 "제조사 후원 연구도 detection rate·false alarm을 보고하지 않는다"고 명시 |
| 4 | **Medtronic Guardian 4 예측 경보 성능** | 공개 데이터 없음 | Pleus et al. 2024 JDST가 "no published performance data exists" 명시 |
| 5 | **Abbott FreeStyle Libre 3 / 3 Plus 예측 경보 유무 및 사양** | 제조사 문서 접근 실패 (dexcom·abbott 문서 URL 404, WebSearch 예산 소진) | Libre **2**에 대해서는 "threshold-based optional alerts only"를 Preechasuk 2024로 확인 |
| 6 | **Toledo-Marín et al. 2023의 분할이 환자 단위인지** | 초록은 "training set was composed of 7% of the data set"라고만 서술. MDPI 전문 403, PMC 매핑 실패 | 분할 방식 불명 상태로 표기. RMSE 24 mg/dL 및 "persistence 대비 유의 개선 없음" 결론만 인용 |
| 7 | **Bertachi et al. 원논문** (Ohio 30분 RMSE 19.33, 야간 저혈당 sens 85% / spec 92%) | Xiong et al. 2025의 인용을 통해서만 확인 | 2차 인용임을 표에 명시 |
| 8 | **SMILE 시험(640G SmartGuard, 고위험 성인) 최종 결과 수치** | 프로토콜 논문만 확인 (De Valk 2018 DTT) | 결과 미인용 |
| 9 | **혈당 "예측" 모델에 대한 Clarke A+B 공식 기준** | 존재하지 않는 것으로 보임. Clarke EGA는 SMBG 기기 평가용(Clarke 1987), ISO 15197은 오차 그리드가 아닌 절대/상대 오차 기준 | 문헌 실측 범위(92–99.6%)를 벤치마크로 제시 |
| 10 | **OpenAPS Data Commons vs OhioT1DM의 혈당 변동성 정량 비교** | 두 데이터셋의 CV/TIR/TBR을 동일 기준으로 비교한 문헌을 찾지 못함 | OpenAPS의 GV 지표는 Cooper 2025 / Shahid 2022로 확인. Ohio 쪽 대응 수치는 미확보 → §6-1의 "쉬운 데이터셋" 우려를 정량화하지 못함 |
| 11 | **30분 horizon에서 sensitivity와 "하루당 거짓경보 횟수"를 함께 보고한 T1D 연구** | 조사 범위에서 발견하지 못함. 유일한 시간 단위 FA 보고는 Prendin 2025(post-bariatric 저혈당, T1D 아님)의 "6일에 1회" | 없음을 §3.1에 사실로 기록 |
