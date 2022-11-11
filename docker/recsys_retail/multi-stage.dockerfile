# First stage - train model
FROM python:3.10.8-slim-buster AS trainer

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

RUN ./scripts/train_save_model.py,\
 -d, https://storage.yandexcloud.net/recsys-retail-input/train.csv.zip,\
 -d, https://storage.yandexcloud.net/recsys-retail-input/item_features.csv,\
 -d, https://storage.yandexcloud.net/recsys-retail-input/user_features.csv, \
 -o, LightGBM_v1, -v

# Second stage - create service and get model from previous stage
FROM python:3.10.8-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

COPY --from=trainer /usr/src/app/models/LightGBM_v1.txt ./models/

RUN useradd --user-group --shell /bin/false recsys-retail  
USER recsys-retail

EXPOSE 8000

ENV MODEL "LightGBM_v1"

CMD ["bash", "start.sh"]