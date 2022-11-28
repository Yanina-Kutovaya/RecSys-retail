FROM python:3.10.8-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

RUN useradd --user-group --shell /bin/false recsys-retail
RUN chown -R recsys-retail models  
USER recsys-retail

CMD ["./scripts/train_save_model.py",\
 "-d1", "https://storage.yandexcloud.net/recsys-retail-input/train.csv.zip",\
 "-d2", "https://storage.yandexcloud.net/recsys-retail-input/item_features.csv",\
 "-d3", "https://storage.yandexcloud.net/recsys-retail-input/user_features.csv", \
 "-o", "baseline_v1", \
"-v"]