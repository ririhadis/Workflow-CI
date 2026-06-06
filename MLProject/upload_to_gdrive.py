import os
import json
import tarfile
import mlflow
import dagshub
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_artifact():
    #kredensial environment variable yang dikirim YAML
    cred_json = os.environ.get('GDRIVE_CREDENTIALS')
    folder_id = os.environ.get('GDRIVE_FOLDER_ID')

    if not cred_json or not folder_id:
        print("Eror: Kredensial atau Folder ID Google Drive tidak ditemukan")

    #menghubungkan mlflow ke repositori dagshub
    print("Menghubungkan ke server repositori DagsHub...")
    dagshub.init(
        repo_owner ='ririhadis',
        repo_name= 'global_energy',
        mlflow= True
    )
    
    #autentikasi ke google drive
    creds_dict = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes =['https://googleapis.com']
    )

    drive_service = build('drive', 'v3', credentials=creds)

    try:
        #cari Run ID terbaru langsung dari server DagsHub
        runs = mlflow.search_runs(experiment_ids=['0'], order_by=['start_time DESC'], max_results=1)
        if runs.empty:
            print("Tidak ada riwayat run MLflow yang ditemukan di DagsHub")
            return

        run_id = runs.iloc[0]['run_id']
        print(f"Mengunduh seluruh artefak dari Run ID DagsHub: {run_id}")

        #unduh seluruh artefak dari dagshub ke lokal sementara
        local_dir = "temp_all_artifacts"
        mlflow.artifacts.download_artifacts(
            artifact_uri = f"runs:/{run_id}",
            dst_path=local_dir
        )

        #kompres seluruh artefak menjadi satu dalam .tar.gz
        archive_name =f"mlflow_artifacts_{run_id}.tar.gz"
        print(f"Mengkompres seluruh artefak menjadi {archive_name}...")
        with tarfile.open(archive_name, "w:gz") as tar:
            tar.add(local_dir, arcname="artifacts")

        #unggah file .tar.gz ke google drive
        file_metadata = {
            'name': archive_name, 'parents': [folder_id]
        }

        media = MediaFileUpload(archive_name, mimetype='application/gzip', resumable=True)
        print("Mengunggah arsip ke Google drive")
        file = drive_service.files().cretae(body=file_metadata, media_body=media, fields='id').execute()
        print("Sukes mengunggah")

        #bersihkan file sampah lokal di runner setelah sukses
        if os.path.exists(archive_name):
            os.remove(archive_name)

    except Exception as e:
        print(f"Proses gagal: {e}")

if __name__ == "__main__":
    get_all_artifacts_and_upload()
