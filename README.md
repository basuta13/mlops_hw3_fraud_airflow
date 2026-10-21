# ДЗ 3: переобучение модели фрода в Airflow

Сделано на основе кода семинара 7. Транзакции идут через Kafka, `fraud_detector` скорит их моделью CatBoost, результаты пишутся в Postgres (`scores`). Airflow поднимается в том же `docker-compose` и в той же сети.

DAG-и:
- `supply_labeled_batch` — каждые 5 минут дописывает в витрину новый месяц;
- `retrain_catboost` — каждые 5 минут переобучает модель и публикует её, если она не хуже текущей;
- `data_quality_checks` — каждые 10 минут проверяет данные и сохраняет HTML-отчёт.

У всех DAG `catchup=False`. Код задач лежит в `pipeline/`, в `dags/` только сборка графа.

## Мои параметры

| | семинар | у меня |
|---|---|---|
| размер батча | 100 000 | 50 000 |
| глубина истории | 3 года | 2 года |
| окно обучения | 3 месяца | 4 месяца |

Окно: 2 месяца train, 1 месяц valid (подбор числа деревьев), самый свежий месяц — test. После подбора модель дообучается на train + valid. Стартовое окно — январь–апрель 2019.

## Запуск

```bash
cp .env.example .env
# в .env указать путь к train.csv: TRAIN_DATA_CSV="/полный/путь/train.csv"
docker compose up --build -d
```

Датасет: https://www.kaggle.com/competitions/teta-ml-1-2025/data (`train.csv`).

- Airflow: http://localhost:8081 (admin / admin)
- Kafka UI: http://localhost:8080
- модели: `./artifacts/models`
- DQ-отчёты: `./artifacts/dq_reports`

Остановить и всё удалить: `docker compose down -v && rm -rf artifacts`

## Как запустить DAG и проверить

1. Открыть Airflow, у `retrain_catboost` нажать ▶ (Trigger DAG). Или из консоли:
   ```bash
   docker compose exec airflow airflow dags trigger retrain_catboost
   ```
2. После прогона в `artifacts/models` появляется новая модель `my_catboost_<версия>.cbm`, а в `latest.txt` записана версия, которая сейчас в проде:
   ```bash
   ls artifacts/models
   cat artifacts/models/latest.txt
   ```
3. Сервис читает `latest.txt` перед каждым сообщением и подхватывает новую модель без перезапуска. Версия модели пишется в таблицу `scores` (и в сообщения топика `scoring`):
   ```bash
   docker compose exec postgres psql -U fraud_user -d fraud_db -c "SELECT model_version, count(*), max(created_at) FROM scores GROUP BY 1 ORDER BY 3;"
   ```
   Новые строки должны иметь ту же версию, что и в `latest.txt`.

## DAG retrain_catboost

```
extract_month_batch → slide_training_window → time_train_test_split → preprocess_dataset
→ fit_catboost → evaluate_prod_model → choose_deployment → publish_model / reject_model
→ advance_training_window
```

- Берётся следующий месяц из витрины, он добавляется в окно, самый старый месяц уходит.
- Split по времени, не случайный.
- Свой препроцессинг (`fraud_detector/src/preprocessing.py`): ночные транзакции, время суток через sin/cos, день недели, логарифм суммы, «круглая» сумма, расстояние и направление до мерчанта, логарифм населения, онлайн-категория. В DAG дополнительно чистится train: битые метки, пропуски, `amount <= 0`, дубли.
- Указатель месяца (`training_state`) двигает только последняя задача, после того как модель сохранена. Если что-то упало раньше, повторный запуск возьмёт тот же месяц, и окно не сдвинется дважды.

### Проверка, что модель стала лучше

Новая модель и текущая прод-модель считают метрики на одном и том же test-месяце. Ни одна из них его не видела, так что сравнение честное.

Метрики:
- **PR-AUC** — основная: фрода около 0.5%, и ROC-AUC почти всегда близок к 1;
- **ROC-AUC** — дополнительная.

Модель публикуется, если `PR-AUC новой >= PR-AUC прода` и ROC-AUC просел не больше чем на 0.005. Иначе срабатывает ветка `reject_model`: модель кладётся в `artifacts/models/rejected/`, `latest.txt` не меняется. Метрики обеих моделей и причина решения сохраняются в `*.metrics.json`.

Исключение: если в проде стартовая модель `initial` из образа, первая модель пайплайна публикуется без сравнения. `initial` обучали вне пайплайна, и её метрики сравнивать нечестно.

## DAG data_quality_checks

Проверяет самый свежий месяц витрины, предыдущий месяц используется как эталон. Отчёт сохраняется в `artifacts/dq_reports/` (рядом с `artifacts/models/`). Последний отчёт — `latest.html`.

Правила:

| № | правило | тип |
|---|---|---|
| 1 | все 18 колонок на месте, числовые колонки числовые | critical |
| 2 | нет пропусков в ключевых полях (время, сумма, target, координаты, категория) | critical |
| 3 | в месяце ≥ 5000 строк, нет пропущенных месяцев | critical |
| 4 | target только 0 или 1 | critical |
| 5 | amount > 0, сумм больше 30 000 не больше 0.1% | critical |
| 6 | координаты в допустимых диапазонах, население > 0 | critical |
| 7 | gender ∈ {F, M}, нет новых категорий `cat_id` | warning |
| 8 | дублей не больше 0.5% | warning |
| 9 | доля фрода от 0.1% до 3% | critical |
| 10 | PSI для amount, cat_id и часа < 0.2 (нет сильного дрейфа) | warning |

Если не прошло хотя бы одно critical-правило, последняя задача `dq_gate` падает, и DAG становится красным.

## Если что-то не работает

- Сайт Airflow первые 1–2 минуты после старта не открывается, это нормально.
- Если после `docker compose restart airflow` сайт не открывается вообще: `docker compose up -d --force-recreate airflow`.
- Образы Kafka/Zookeeper обновлены до `7.6.1`: на `7.3.0` Zookeeper падал на новом Docker Desktop.
