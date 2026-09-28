"""
deploy_endpoint.py - Deploy model ke SageMaker real-time endpoint.
"""

from __future__ import annotations

import json

import boto3
import sagemaker
from sagemaker.sklearn.model import Model

BUCKET_NAME = "credit-score-2802545466"
ENDPOINT_NAME = "credit-score-endpoint"

S3_MODEL_URI = f"s3://{BUCKET_NAME}/model/model.tar.gz"

# Test payload: satu customer dengan profil standar
TEST_PAYLOAD = {
    "Age": 30,
    "Annual_Income": 50000.0,
    "Monthly_Inhand_Salary": 4000.0,
    "Num_Bank_Accounts": 2,
    "Num_Credit_Card": 3,
    "Interest_Rate": 12,
    "Num_of_Loan": 1,
    "Delay_from_due_date": 5,
    "Num_of_Delayed_Payment": 2,
    "Changed_Credit_Limit": 10.0,
    "Num_Credit_Inquiries": 1,
    "Credit_Mix": "Bad",
    "Outstanding_Debt": 1500.0,
    "Credit_Utilization_Ratio": 30.0,
    "Credit_History_Age": "22 Years and 4 Months",
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 300.0,
    "Amount_invested_monthly": 100.0,
    "Payment_Behaviour": "Low_spent_Small_value_payments",
    "Monthly_Balance": 500.0,
    "Type_of_Loan": "Personal Loan and Home Loan",
    "Occupation": "Scientist",
    "Month": "January",
}


def main() -> None:

    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    print("=" * 60)
    print(f"Role ARN  : {role}")
    print(f"Model URI : {S3_MODEL_URI}")
    print(f"Endpoint  : {ENDPOINT_NAME}")
    print("=" * 60)
    print("Deploying real-time endpoint...")
    print("(Estimasi waktu: 5-10 menit)")

    model = Model(
    model_data=S3_MODEL_URI,
    role=role,
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=session.boto_region_name,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.large",
    ),
    entry_point="inference_sagemaker.py",
    sagemaker_session=session,
)

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME,
    )

    print("\nEndpoint berhasil dibuat. Menjalankan test...")

    response = predictor.predict(json.dumps(TEST_PAYLOAD))

    print("\nCredit test response:")
    if isinstance(response, (bytes, str)):
        parsed = json.loads(response)
        print(json.dumps(parsed, indent=2))
    else:
        print(response)

    print("\nDeploy selesai.")
    print(f"Endpoint name: {ENDPOINT_NAME}")
    print("\nLangkah berikutnya:")
    print("  1. Cek endpoint di AWS Console -> SageMaker AI -> Inference -> Endpoints")
    print("  2. Jalankan Streamlit di EC2 untuk deployment publik")


if __name__ == "__main__":
    main()
