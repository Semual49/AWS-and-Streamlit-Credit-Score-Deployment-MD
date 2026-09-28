# Panduan Deployment AWS - Credit Score Classifier

Letakkan folder `aws/` di root project, setara dengan `app/`, `src/`, `notebooks/`.

```
credit_score_project/
├── app/
├── aws/
│   ├── data/                    <- taruh data_C.csv di sini
│   ├── model_artifact/          <- dibuat otomatis saat training
│   ├── inference_sagemaker.py
│   ├── pipeline_sagemaker.py
│   ├── package_model.py
│   ├── upload_to_s3.py
│   ├── deploy_endpoint.py
│   ├── streamlit_ec2.py
│   ├── requirements_aws.txt
│   └── README_AWS.md
├── src/
├── data/
└── ...
```

---

## Sebelum Mulai

Taruh `data_C.csv` di `aws/data/data_C.csv` dan tentukan nama bucket.

---

## Step 1: Buka AWS Academy dan SageMaker AI

1. Login ke AWS Academy Learner Lab, klik **Start Lab**, tunggu indikator AWS jadi hijau.
2. Klik **AWS** untuk masuk ke AWS Management Console.
3. Pastikan region di pojok kanan atas sudah **US East (N. Virginia) us-east-1**.
4. Di search bar, ketik **SageMaker**, klik **Amazon SageMaker AI**.
5. Di sidebar kiri, klik **Notebooks**.
6. Klik **Create notebook instance**.
7. Isi nama bebas, tipe `ml.m5.xlarge`, IAM role pilih **LabRole**.
8. Klik **Create**. Tunggu status **InService**.

---

## Step 2: Upload Project ke SageMaker

9. Klik **Open JupyterLab** di samping nama instance.
10. Di JupyterLab, klik **Terminal**.
11. Ketik:

```bash
cd /home/ec2-user/SageMaker
```

12. Zip seluruh folder `credit_score_project` di komputer lokal:
    - Klik kanan folder -> **Send to > Compressed (zipped) folder**
13. Drag-and-drop file zip ke panel file browser JupyterLab sebelah kiri.
14. Tunggu upload selesai, lalu extract:

```bash
unzip credit_score_project.zip
cd credit_score_project
```

15. Verifikasi file aws ada:

```bash
ls aws/
```


---

## Step 3: Setup Virtual Environment

```bash
sudo mkdir /my_venv_global
sudo chown ec2-user:ec2-user /my_venv_global
python3 -m venv /my_venv_global/my_venv
source /my_venv_global/my_venv/bin/activate
pip install -r aws/requirements_aws.txt
```

Kalau ada konflik numpy atau sklearn:

```bash
pip install --force-reinstall numpy==1.24.3 scikit-learn==1.2.1
```

Verifikasi:

```bash
pip show sagemaker
```


---

## Step 4: Upload Data dan Training Pipeline

16. Di panel file browser JupyterLab, navigasi ke `credit_score_project/aws/data/`.
17. Drag-and-drop `data_C.csv` dari komputer lokal ke folder tersebut.
18. Kembali ke terminal, jalankan training:

```bash
python aws/pipeline_sagemaker.py
```

Tunggu selesai, bisa 10-25 menit. Output akhir yang benar:

```
"best_model": "random_forest",
"test_accuracy": 0.73xx,
"test_f1_weighted": 0.73xx

Pipeline selesai.
Model tersimpan di: .../aws/model_artifact/model_best.pkl
```


---

## Step 5: Package Model

```bash
python aws/package_model.py
```

Output yang benar:

```
Berhasil. Isi model.tar.gz:
  model_best.pkl  (xxx bytes)
  inference_sagemaker.py  (xxx bytes)
```


---

## Step 6: Upload ke S3

Edit `aws/upload_to_s3.py` langsung di JupyterLab, ganti `BUCKET_NAME`. Simpan.

```bash
python aws/upload_to_s3.py
```

Output yang benar:

```
make_bucket: credit-score-namakamu
upload: aws/model_artifact/model.tar.gz to s3://credit-score-namakamu/model/model.tar.gz
Upload berhasil.
```


Verifikasi di AWS Console -> **S3**. Pastikan bucket muncul di daftar.


---

## Step 7: Deploy Endpoint

Edit `aws/deploy_endpoint.py` di JupyterLab, ganti `BUCKET_NAME` sama dengan yang di upload_to_s3.py. Simpan.

Pastikan kamu berada di folder `credit_score_project` sebelum jalankan:

```bash
python aws/deploy_endpoint.py
```

Tunggu 5-10 menit. Output akhir yang benar:

```
Deploying real-time endpoint...
Endpoint berhasil dibuat. Menjalankan test...
Credit test response:
{"predictions": ["Standard"], "probabilities": [...], "labels": ["Good", "Poor", "Standard"]}
```

Kalau muncul error `FileNotFoundError: inference_sagemaker.py`, itu artinya kamu tidak berada
di folder `credit_score_project`. Jalankan `cd /home/ec2-user/SageMaker/credit_score_project`
lalu ulangi.


---

## Step 8: Verifikasi Endpoint di Console

Di AWS Console -> **SageMaker AI** -> sidebar kiri bagian **Deployments & inference** -> **Endpoints**.

Pastikan `credit-score-endpoint` status **InService**.


Klik nama endpoint untuk lihat detail.


---

## Step 9: Cek CloudWatch Logs

AWS Console -> **CloudWatch** -> **Log management** -> **Log groups**.

Cari `/aws/sagemaker/Endpoints/credit-score-endpoint`, klik log stream terbaru.

Pastikan ada baris:
```
[ENDPOINT LOG] Model loaded: random_forest
[ENDPOINT LOG] Feature count: xx
```


---

## Step 10: Setup EC2 untuk Streamlit Publik

Di AWS Console -> **EC2** -> **Launch Instances**.

Konfigurasi:
- Name: `credit-app-ec2`
- AMI: Amazon Linux 2023
- Instance type: `t3.micro`
- Key pair: buat baru, download `.pem`
- Security group: tambah inbound rule port `8501`, source `0.0.0.0/0`

Klik **Launch instance**. Tunggu status **Running** dan **3/3 checks passed**.


---

## Step 11: SSH ke EC2

Buka PowerShell di komputer lokal:

```powershell
cd $HOME\Downloads
ssh -i "credit-app-key.pem" ec2-user@IP_PUBLIC_EC2
```

Ketik `yes` saat muncul konfirmasi fingerprint.

---

## Step 12: Setup dan Jalankan Streamlit di EC2

Di dalam EC2:

```bash
pip3 install streamlit scikit-learn==1.2.1 pandas numpy joblib lightgbm
pip3 install numpy==1.24.3
```

Upload zip project dari PowerShell baru (jangan tutup yang sudah SSH):

```powershell
scp -i "credit-app-key.pem" "PATH_LENGKAP\credit_score_project.zip" ec2-user@IP_PUBLIC_EC2:~/
```

Kembali ke terminal yang sudah SSH, extract dan setup:

```bash
unzip credit_score_project.zip
cd credit_score_project
mkdir -p aws/model_artifact
```

Download model dari S3. Di EC2 tidak ada akses S3 otomatis, jadi perlu configure dulu.
Ambil credentials dari AWS Academy Learner Lab (klik **AWS Details** di halaman lab):

```bash
aws configure
```

Isi AWS Access Key ID, Secret Access Key, Session Token, dan region `us-east-1`
sesuai yang tertera di AWS Details. Lalu:

```bash
aws s3 cp s3://credit-score-namakamu/model/model.tar.gz aws/model_artifact/
cd aws/model_artifact
tar -xzf model.tar.gz
cd ../..
```

Verifikasi:

```bash
ls aws/model_artifact/
```

Harus ada `model_best.pkl` dan `inference_sagemaker.py`.

Jalankan Streamlit dari dalam folder `aws/`:

```bash
cd /home/ec2-user/credit_score_project/aws
streamlit run streamlit_ec2.py
```

Output yang benar:

```
You can now view your Streamlit app in your browser.
  External URL: http://IP_PUBLIC_EC2:8501
```


---

## Step 13: Verifikasi Aplikasi di Browser

Buka browser, ketik:

```
http://IP_PUBLIC_EC2:8501
```


---

## Step 14: Test Case

Buka tab **Manual Entry**. Lakukan 3 test berikut.

### STANDARD
- Annual Income: 50000, Outstanding Debt: 1500, Num Delayed Payment: 2
- Credit Mix: Standard, Interest Rate: 12, Credit Utilization Ratio: 30
- Payment Behaviour: Low_spent_Small_value_payments, Monthly Balance: 500


### GOOD
- Annual Income: 122000, Outstanding Debt: 200, Num Delayed Payment: 0
- Credit Mix: Good, Interest Rate: 1, Credit Utilization Ratio: 10
- Payment Behaviour: Low_spent_Small_value_payments, Amount Invested Monthly: 5000


### POOR
- Annual Income: 100, Outstanding Debt: 200000, Num Delayed Payment: 10
- Credit Mix: Bad, Interest Rate: 30, Num Credit Inquiries: 10
- Payment Behaviour: Low_spent_Small_value_payments, Num of Loan: 6


Bisa juga test via tab **CSV Upload** dengan upload `test_cases.csv`.


---

## Step 15: Cleanup

Setelah semua selesai, hapus semua resource agar tidak membuang credit lab.

1. **Hapus SageMaker Endpoint**: SageMaker AI -> Deployments & inference -> Endpoints
   -> pilih endpoint -> **Actions -> Delete**.

2. **Hapus SageMaker Notebook Instance**: SageMaker AI -> Notebooks
   -> pilih instance -> **Actions -> Stop**, tunggu Stopped, lalu **Actions -> Delete**.

3. **Kosongkan dan hapus S3 Bucket**: S3 -> klik nama bucket -> **Empty** -> konfirmasi.
   Lalu kembali ke daftar bucket -> pilih bucket -> **Delete** -> konfirmasi.

4. **Terminate EC2**: EC2 -> Instances -> pilih instance -> **Instance state -> Terminate**.

5. Kembali ke AWS Academy Learner Lab, klik **End Lab**.