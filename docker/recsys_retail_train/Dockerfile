FROM python:3.8.11-slim

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
 "-d", "https://storage.yandexcloud.net/recsys-retail-input/train.csv.zip",\
 "-d", "https://storage.yandexcloud.net/recsys-retail-input/item_features.csv",\
 "-d", "https://storage.yandexcloud.net/recsys-retail-input/user_features.csv", \
 "-o", "LightGBM_v1", \
"-v"]