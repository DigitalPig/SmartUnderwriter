FROM heroku/miniconda

# COPY necessary files inside
ADD ./app /opt/app
WORKDIR /opt/app
COPY environment.yml /opt/app
COPY run.py /opt/app
COPY start.sh /opt/app

RUN conda env create -f environment.yml

CMD start.sh