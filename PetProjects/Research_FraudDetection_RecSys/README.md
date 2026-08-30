# Fraud Detection via Recommender Systems Methods

Магистерская курсовая работа, ВШЭ, ПМИ, ОП "Финансовые технологии и анализ данных"  
**Тема:** Оценка эффективности алгоритмов рекомендательных систем в задаче обнаружения мошеннических транзакций  
**Руководитель:** Соколовский Е.И.

---

## Описание

Исследование сравнивает девять методов из области рекомендательных систем, адаптированных для детекции мошеннических транзакций по кредитным картам. Эксперименты проводятся на датасете Sparkov (Kaggle), содержащем 1.85M синтетических транзакций.

Методы разбиты на четыре категории:
| Категория | Метод | Источник |
|---|---|---|
| Baseline | Logistic Regression, LightGBM (3 конфигурации признаков) | — |
| Матричная факторизация | CTKT, GiPM3F | Cui et al. 2021; Yang et al. 2018 |
| Графовые | GraphRfi, EthAegis | Zhang et al. SIGIR 2020; Jain & Tripathy AISec 2025 |
| Последовательные | SPADE, CLUE, NHFM | Kim et al. 2022; Wang et al. ECML PKDD 2017; Xi et al. SIGIR 2020 |

Baseline-бустинг представлен тремя конфигурациями признаков для разделения
вкладов статического и временно́го сигнала: LightGBM-32 (статические
агрегаты), LightGBM-velocity (только 12 каузальных временны́х признаков)
и LightGBM-44 (комбинация) — при идентичных гиперпараметрах.

---

## Результаты
| Метод | ROC-AUC | PR-AUC [95% CI] | P@100 | P@500 |
|---|---|---|---|---|
| **LightGBM-44** (32+12 velocity) | **0.995** | **0.885** [0.874; 0.896] | 1.000 | 1.000 |
| **CLUE** | 0.991 | 0.809 [0.793; 0.824] | 1.000 | 0.998 |
| LightGBM-velocity (12) | 0.969 | 0.733 [0.715; 0.751] | 1.000 | 0.988 |
| NHFM | 0.983 | 0.708 [0.688; 0.726] | 0.990 | 0.978 |
| LightGBM-32 | 0.903 | 0.546 [0.525; 0.567] | 1.000 | 0.950 |
| Logistic Regression | 0.931 | 0.132 [0.123; 0.142] | 0.000 | 0.018 |
| GraphRfi | 0.545 | 0.084 [0.072; 0.095] | 1.000 | 0.344 |
| SPADE | 0.619 | 0.075 [0.063; 0.089] | 0.650 | 0.388 |
| CTKT | 0.894 | 0.068 [0.062; 0.076] | 0.260 | 0.166 |
| EthAegis | 0.848 | 0.018 [0.017; 0.019] | 0.050 | 0.022 |
| GiPM3F | 0.536 | 0.011 [0.010; 0.013] | 0.020 | 0.040 |

Решающим фактором качества является доступ модели к временно́й динамике
поведения держателя карты, а не класс архитектуры сам по себе. Это показано
двумя независимыми способами: (1) внутри одного класса моделей — 12
velocity-признаков без единого статического (PR-AUC 0.733) превосходят 32
статических (0.546), а комбинация (0.885) даёт лучший результат
эксперимента; (2) между классами — нейросетевое моделирование порядка
событий (CLUE, 0.809) извлекает из пяти сырых атрибутов больше, чем бустинг
из двенадцати сконструированных временны́х признаков. Все ключевые попарные
различия PR-AUC значимы по парному бутстрэпу (1000 общих ресэмплов);
LightGBM-44 − CLUE = [+0.070; +0.099].

---
## Структура проекта

```
fraud_detection/
├── data/
│   ├── raw/
│   │   ├── fraudTrain.csv          # Sparkov train (2019, 1.3M транзакций)
│   │   └── fraudTest.csv           # Sparkov test  (2020, 556K транзакций)
│   └── processed/
│       ├── train_features.csv      # Обработанные признаки train (32 признака)
│       ├── test_features.csv       # Обработанные признаки test
│       └── target_encoding_maps.json  # Карты target encoding
├── notebooks/
│   ├── notebook_1_eda_preprocessing.ipynb   # EDA и feature engineering
│   ├── notebook_2_baseline_ml.ipynb         # LR, LightGBM + velocity-эксперимент
│   ├── notebook_3_factorization.ipynb       # CTKT и GiPM3F
│   ├── notebook_4_graph.ipynb               # GraphRfi и EthAegis
│   ├── notebook_5_sequential.ipynb          # SPADE, CLUE и NHFM
│   └── notebook_6_evaluation.ipynb          # Финальное сравнение
├── models/
│   ├── lgbm_model.pkl              # LightGBM-32 (статические признаки)
│   ├── lgbm_velocity_model.pkl     # LightGBM-velocity (12 временных)
│   ├── lgbm_44_model.pkl           # LightGBM-44 (комбинация)
│   ├── lgbm_best_params.json       # Лучшие параметры Optuna
│   ├── velocity_cols.json          # Состав velocity-признаков
│   └── lr_pipeline.pkl             # Pipeline логистической регрессии
├── results/
│   ├── scores_*.csv                # Скоры каждой модели на test
│   ├── metrics_all.csv             # Сводная таблица метрик (11 моделей)
│   ├── metrics_with_ci.csv         # Метрики с бутстрэп-интервалами
│   ├── asymmetric_metrics.csv      # F-beta и денежно-взвешенный Recall@K
│   ├── pr_curves.png               # PR-кривые по категориям
│   ├── ranking_metrics.png         # Precision@K и NDCG@K
│   ├── metrics_barplot.png         # Барчарт основных метрик
│   └── publication/                # Монохромные рисунки для статьи (EPS/TIFF)
└── src/                            # (зарезервировано)
```

---
## Датасет
**Sparkov Credit Card Fraud Detection** ([Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection))
- 1,296,675 транзакций в train (2019), 555,719 в test (2020)
- 1,000 держателей карт, 800 мерчантов
- Доля фрода: 0.58% train / 0.39% test
- Синтетически сгенерирован с помощью Sparkov Data Generation
> **Важно:** файлы датасета не включены в репозиторий из-за размера.  
> Скачайте `fraudTrain.csv` и `fraudTest.csv` с Kaggle и поместите в `data/raw/`.
---

## Окружение
**Требования:** Python 3.10, macOS (Apple Silicon M1+) или Linux
```bash
# Создание окружения
conda create -n fraud python=3.10
conda activate fraud
# Основные зависимости
pip install torch==2.4.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
pip install torch_scatter \
    -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch_geometric==2.8.0
pip install lightgbm optuna
pip install pandas numpy scikit-learn matplotlib seaborn
pip install networkx jupyter
```
> **Примечание для Apple Silicon:** `torch_scatter` не поддерживает MPS,
> все графовые вычисления выполняются на CPU. Это учтено в коде.

---

## Запуск
Ноутбуки выполняются последовательно — каждый следующий использует
артефакты предыдущего:
```bash
jupyter lab
```
| Шаг | Ноутбук | Что делает | Время |
|---|---|---|---|
| 1 | `notebook_1_eda_preprocessing` | EDA, feature engineering, сохранение признаков | ~5 мин |
| 2 | `notebook_2_baseline_ml` | LR + LightGBM (Optuna) + velocity-эксперимент | ~20 мин |
| 3 | `notebook_3_factorization` | CTKT + GiPM3F | ~30 мин |
| 4 | `notebook_4_graph` | GraphRfi + EthAegis | ~60 мин |
| 5 | `notebook_5_sequential` | SPADE + CLUE + NHFM | ~60 мин |
| 6 | `notebook_6_evaluation` | Сравнение, бутстрэп-интервалы, визуализации | ~10 мин |

При повторных прогонах Notebook 2 поиск Optuna пропускается: гиперпараметры
и обученные модели загружаются из `models/`.

---

## Метрики оценки
- **PR-AUC** — качество при сильном дисбалансе классов (основная метрика),
  с 95% бутстрэп-интервалами и парным бутстрэпом разностей для ключевых пар
- **ROC-AUC** — общее качество разделения классов
- **Precision@K / NDCG@K** — качество ранжирования в топе очереди алертов
- **F1 / F-beta** — баланс точности и полноты при оптимальном пороге;
  F2 и F0.5 — проверка устойчивости к асимметрии стоимости ошибок
- **Денежно-взвешенный Recall@K** — доля предотвращённого ущерба в топ-K
---
## Ключевые выводы
1. Решающий фактор качества — доступ модели к временно́й динамике поведения
   держателя: 12 каузальных velocity-признаков (PR-AUC=0.733) превосходят
   32 статических (0.546), комбинация LightGBM-44 (0.885) — лучший
   результат эксперимента, значимо опережающий CLUE ([+0.070; +0.099]).
2. Последовательные нейросетевые методы — сильнейшие среди подходов без
   конструирования признаков: CLUE достигает PR-AUC=0.809 на пяти сырых
   дискретизированных атрибутах, значимо опережая бустинг на двенадцати
   сконструированных временны́х признаках.
3. Носителем временно́го сигнала выступает отклонение от скользящего
   персонального профиля (`amt_roll_z`), а не частотные счётчики: фрод
   в Sparkov — не высокочастотный «прогон» карты.
4. Ранжирование методов устойчиво к асимметрии стоимости ошибок
   (корреляции рангов F1/F2 = 0.991, F1/F0.5 = 0.936); в терминах
   предотвращённого ущерба топ-500 очереди накрывает до 35–38% суммарных
   мошеннических средств при 23% транзакций.
5. Графовые и матричные методы ограничены структурой синтетического
   датасета: 77.5% держателей имеют хотя бы одну мошенническую транзакцию
   в train, что делает узловые метки малоинформативными.
6. SPADE — единственный ненейросетевой метод — ограничен отсутствием
   устойчивых персональных паттернов в синтетических данных и холодным
   стартом (7.6% фрода теста приходится на держателей без истории).
---
## Литература
- Zhang et al. *GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Detection*. SIGIR 2020.
- Xi et al. *Neural Hierarchical Factorization Machines for User's Event Sequence Analysis*. SIGIR 2020.
- Wang et al. *Session-Based Fraud Detection in Online E-Commerce Transactions Using Recurrent Neural Networks*. ECML PKDD 2017.
- Jain & Tripathy. *EthAegis: Featured Graph Based Fraud Detection in Ethereum Transactions*. AISec 2025.
- Kim et al. *Sequential Pattern Mining Approach for Personalized Fraudulent Transaction Detection in Online Banking*. Sustainability 2022.
- Cui et al. *A Credible Individual Behavior Profiling Method for Online Payment Fraud Detection*. DSDE 2021.
- Yang et al. *Evaluating Prediction Error for Anomaly Detection by Exploiting Matrix Factorization in Rating Systems*. IEEE Access 2018.
