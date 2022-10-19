FROM python:3.8.11

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .recsys-retail

RUN useradd --user-group --shell /bin/false   
USER recsys-retail

EXPOSE 8000

ENV MODEL "LightGBM_v1"

CMD ["bash", "start.sh"]

