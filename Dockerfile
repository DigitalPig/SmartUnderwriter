FROM heroku/miniconda

# COPY necessary files inside
ADD ./app /opt/web/app
WORKDIR /opt/web/
COPY environment.yml /opt/web
COPY run.py /opt/web
COPY config.py /opt/web
COPY start.sh /opt/web

RUN conda update -y conda && \
    conda env create -f environment.yml

CMD bash ./start.sh