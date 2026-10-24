# SWEET.TV: воспроизводимый рекомендатель телепрограмм

Открытый репозиторий: [https://github.com/ulianakazak/sweettv-recommender-exam](https://github.com/ulianakazak/sweettv-recommender-exam). Репозиторий создан; загрузка файлов выполняется, проверка удалённого коммита и совпадения SHA-256 ещё ожидается. Эта локальная версия не утверждает завершённую верификацию публикации.

Итоговый `artifacts/best_local_v4_submission.csv` содержит 1260 пользователей, по пять уникальных ненулевых передач. SHA-256: `0dc3c8601b01ba093c302feedd257cf11bcdad8106ac9fad53ad6b58b51a03ab`.
Сохранённая проверенная запись оркестратора подтверждает private **0.41586**, public **0.42282**, статус Kaggle `Complete (after deadline)`. Это отдельное внешнее подтверждение после локального выбора; оно не использовалось для настройки. Новых внешних действий этот репозиторий не делает.

## Что включено

Полные исходники подготовки и моделей: train-popularity, sparse CBF, собственная implicit ALS (Hu–Koren–Volinsky), V1 frequency/decay, V2 mass-preserving power, V3 user-neighbour CF/day caps и выбранный V4 partial+exposure. `src/sweettv/*.py` скопированы без изменения модели. Новый `scripts/run_local.py` переносит только CLI и пути. Версии библиотек, seed 42 / threads 1 и фиксированные конфигурации сохранены в `requirements.lock.txt` и `configs/`. Тесты используют синтетические ID.

В `evidence/` — агрегатные результаты и excerpts фактически выполненных независимых проверок, без per-user histories. В папке нет raw CSV, caches, credentials, browser states и полных историй AI-чатов. Они не нужны для просмотра кода; raw нужны для пересборки признаков. Обязательный submission CSV содержит реальные user IDs и включён отдельно по условию задания.

## Установка и быстрый smoke check

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/run_local.py --help
.venv/bin/python scripts/run_local.py verify
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Исходный runtime: Python3.12.14. Seed 42 и один вычислительный поток фиксированы; скрытый Kaggle evaluator локально не воспроизводится. `verify` проверяет точные байты CSV и формат; исходная независимая проверка membership в candidate pool отражена в aggregate evidence.

## Подготовка из внешнего защищённого датасета

Поместите исходные файлы, перечисленные в `docs/DATA.md`, в приватную папку. `--raw-dir` может быть относительным или абсолютным, Mac-пути в конфигурации не зашиты. Запускайте в новой копии/свежих output directories: guards запрещают случайную перезапись.

```sh
.venv/bin/python scripts/run_local.py prepare --raw-dir data/raw
.venv/bin/python scripts/run_local.py baselines
.venv/bin/python scripts/run_local.py prepare-v4 --raw-dir data/raw
.venv/bin/python scripts/run_local.py evaluate-v4
.venv/bin/python scripts/run_local.py fit-final
```

`prepare` выполняет same-channel interval union и join с полным исходным duration. Qualified airing: watched>=.8 full duration; ID0 исключён; окна half-open; каждый airing считается один раз. `baselines` выполняет исходный tune/main/final конвейер popularity/CBF/MF из `sweettv.evaluate`, сохраняя его оригинальные метрики и выдачи. `prepare-v4` строит все шесть окон без обучения; `evaluate-v4` повторяет фиксированные 26 конфигураций на трёх development folds; `fit-final` применяет уже выбранный `combo_p2_raw_b025` на полном TRAIN, не выбирая стратегию по будущим данным. Generated caches, per-user predictions и новые результаты попадут только в игнорируемые `work/` и `outputs/`.

Для точного frozen replay, включая диагностические окна, доступен отдельный **локальный** полный reproducibility archive: внутри него вложен неизменный проверенный V4 cached-replay ZIP 185 MB с точными frozen inputs, original snapshots и полными independent checks. Он содержит пользовательские caches и не предназначен для GitHub. Инструкция вложенного архива требует новую извлечённую копию, перенос completed outputs в evidence и fresh freeze/run/finish; raw CSV для этого cached replay не нужны. UTC/runtime поля при повторении меняются, rankings/AP и финальный CSV должны совпасть. Публичная папка не выдаёт отсутствие приватных caches за автономный exact replay без внешнего датасета.

## Методы и честное сравнение

Original common holdout: TRAIN09.03–18.05, target18.05–27.07, те же 1260 пользователей и 6384 available IDs, MAP@5 с denominator min(5, truth size), пустая truth даёт0. Truth — top5 qualified airing frequencies DESC, ID ASC. Все методы возвращают пять уникальных IDs, допускают повторы истории и используют одинаковый порядок train-popularity fallback. Детали невыровнены искусственно: 18 true-cold users совпадают, а V4 дополнительно falls back у 3 пользователей с qualified history только вне доступного pool; оригинальные CBF/MF могут использовать learned profile. Сохранённые результаты не переписаны ради косметического равенства. Таблица — `evidence/common_holdout_original.csv`.

V4 выбрана на rolling targets по 28 дней: MAP 0.358208/0.271802/0.342980, mean 0.324330. Она добавляет weak coverage**2 (60+s, .05<=coverage<.8), нормирует weak напрямую L1, смешивает с sqrt-qualified profile при исходной confidence mass и prior 30; затем умножает score на clipped timetable exposure ratio**.25. Будущие расписания используются структурно, future viewing/metadata — нет. Частота квалифицированных эфиров и published target не изменены.

CBF применяет train-only DictVectorizer+TF–IDF и cosine profile; implicit MF — собственный переносимый ALS с confidence1+alpha*log(1+count), так как исходная установка implicit не собралась на arm64. Это реальный engineering failure, а не причина скрыть результат MF. Все первоначальные baseline scores сохранены.

Все окна уже исследовались в разработке (в том числе150 V2, 10 V3 и 24 V4 конфигурации); это не untouched test. Four-week rolling targets короче финального84-day горизонта. Новое правило V4 не меняет прежний отрицательный вердикт V3. Проверка кода/источников выполнена AI; личная ручная проверка студентом этим не подтверждается. Реальные первичные источники указаны в `REFERENCES.md`; association rules/PPR упомянуты как неоценённые альтернативы.
