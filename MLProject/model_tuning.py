import itertools
import argparse
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import dagshub
import matplotlib.pyplot as plt

#tracking adaptif
#if "MLFLOW_TRACKING_URI" in os.environ:
#    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
#else:
#    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

parser = argparse.ArgumentParser()
parser.add_init =True
parser.add_argument('--dataset', type=str, default= 'data_preprocessing.csv')
parser.add_argument('--init', type=str, default= 'k-means++')
parser.add_argument('--max_iter', type=int, default= 300)
parser.add_argument('--n_clusters', type=int, default= 4)
args, unknown = parser.parse_known_args()

#Menghubungkan ke dagshub
dagshub.init(repo_owner="ririhadis", repo_name="global_energy", mlflow=True)

#Membuat MLflow Eksperiment
#mlflow.set_experiment("Global Renewable Energy Transition from 200 to 2025")

df = pd.read_csv(args.dataset)

fitur = ['income_group', 'population', 'gdp_usd', 'total_electricity_generation_twh', 'electricity_demand_twh',
    'solar_electricity_twh', 'wind_electricity_twh', 'renewables_electricity_twh', 'hydro_electricity_twh', 'nuclear_electricity_twh',
    'fossil_electricity_twh', 'solar_share_pct', 'wind_share_pct', 'renewables_share_pct', 'fossil_share_pct', 'low_carbon_share_pct',
    'carbon_intensity_gco2_kwh', 'co2_saved_solar_wind_mt', 'solar_yoy_growth_pct', 'wind_yoy_growth_pct', 'renewables_yoy_growth_pct',
    'difference', 'indicator']
df_model = df[fitur]

#Log dataset manual
mlflow_dataset = mlflow.data.from_pandas(
    df_model, name='renewable_energy_features')

#parameter tunig
#with mlflow.start_run(nested=True):
    #log informasi dataset kedalam run saat ini
mlflow.log_input(mlflow_dataset, context='training')
    
param_grid={
  'n_clusters' : range(2,6),
  'init' : ['k-means++', 'random'],
  'max_iter': [100, 300]
}

keys, values = zip(*param_grid.items())
kombinasi_param = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
best_score = -1
best_config = None
best_model = None
       
print("Memulai proses parameter tuning")
for config in kombinasi_param:
   model_kmeans = KMeans(
      n_clusters=config['n_clusters'],
      init=config['init'],
      max_iter=config['max_iter'],
      n_init=10,
      random_state=42
   )
        
   labels = model_kmeans.fit_predict(df_model)
   score = silhouette_score(df_model, labels)
   print(f"Parameter: {config} || Silhoutte Score: {score:.4f}")
        
   if score > best_score:
     best_score = score
     best_config = config
    
print("\nHasil Tuning Terbaik")
print(f"Parameter Terbaik: {best_config}")
print(f"Skor Silhouette Tertinggi: {best_score:.4f}")
    
best_model = KMeans(
   n_clusters=best_config['n_clusters'],
   init=best_config['init'],
   max_iter=best_config['max_iter'],
   n_init=10,
   random_state=42
)

best_model.fit(df_model)
    
#Mencatat parameter dan metrik dari model terbaik ke mlflow dashboard
for param_name, param_val in best_config.items():
    mlflow.log_param(f"best_{param_name}", param_val)
mlflow.log_metric("best_silhouette_score", best_score)

inertia_value = model_kmeans.inertia_
mlflow.log_metric("inertia ", inertia_value)
    
#Log manual parameter dan metric manual
#Mengambil sampel data dan hasil prediksi untuk membuat skema input-output
sample_input = df_model.head(5)
sample_output = best_model.predict(sample_input)
signature = infer_signature(sample_input, sample_output)

#Log manual artifact (model & file)
#log model KMeans beserta isi signature-nya ke MLflow Artifacts
mlflow.sklearn.log_model(
    sk_model = best_model,
    artifact_path="kmeans_model",
    signature=signature,
    registered_model_name="KMeans_Renewable_Energy_Model")

#log artifact tambahan 1
plt.figure(figsize=(8,5))
plt.scatter(df['country'], df['year'], c=labels, cmap='inferno')
plt.title("Model Tuning Cluster Result")
plt.xlabel("Country")
plt.ylabel("Year")
#simpan grafik ke lokal sementara lalu upload ke mlflow/DagsHub
graph_path = "global_tuning.png"
plt.savefig(graph_path)
mlflow.log_artifact(graph_path)
    
#Log manual artifact untuk dataset
df_model.to_csv("data_training_used.csv", index=False)
mlflow.log_artifact("data_training_used.csv", artifact_path = "dataset")
    
