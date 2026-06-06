FROM python:3.12-slim

RUN pip install --no-cache-dir setuptools

RUN pip install --no-cache-dir\
   mlflow==2.19.0 \
   cloudpickle==3.1.2 \
   numpy==2.4.4 \
   pandas==2.3.3 \
   psutil==7.0.0 \
   pyarrow==18.1.0 \
   scikit-learn==1.8.0 \
   scipy==1.17.1 \

WORKDIR /global_energy_clustering

#menyalin semua file proyek ke dalam container
COPY . /global_energy_clustering

#perintah untuk menjalankan tuning saat container di jalankan
CMD ["pyhton", "model_tuning.py"]