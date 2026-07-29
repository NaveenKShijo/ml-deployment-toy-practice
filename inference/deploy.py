import boto3
import os
import time
import sys

def deploy():
    sagemaker_client = boto3.client('sagemaker')

    # Get deployment parameters from environment variables
    model_name = os.environ.get('MODEL_NAME', f'practice-model-1')
    role_arn = os.environ['SAGEMAKER_ROLE']
    image_uri = os.environ['IMAGE_URI']
    model_data_url = os.environ['MODEL_DATA_URL']
    endpoint_name = os.environ.get('ENDPOINT_NAME', 'practice-model-endpoint')

    print(f"Deploying model: {model_name}")
    print(f"Image URI: {image_uri}")
    print(f"Model Data URL: {model_data_url}")

    # 1. Create SageMaker Model
    print("Creating SageMaker Model...")
    try:
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url
            },
            ExecutionRoleArn=role_arn
        )
        print(f"Model {model_name} created successfully.")
    except sagemaker_client.exceptions.ClientError as e:
        if 'ValidationException' in str(e) and 'already exists' in str(e):
            print(f"Model {model_name} already exists. Skipping creation.")
        else:
            raise

    # 2. Create Endpoint Configuration
    # SageMaker has a 63-character limit for EndpointConfigName.
    # We truncate model_name to leave room for '-config-' and the 10-digit timestamp.
    short_model_name = model_name[:44]
    endpoint_config_name = f"{short_model_name}-config-{int(time.time())}"
    print(f"Creating Endpoint Configuration: {endpoint_config_name}...")
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.t2.medium',  # Adjust instance type as needed
            }
        ]
    )
    print(f"Endpoint Configuration {endpoint_config_name} created successfully.")

    # 3. Create or Update Endpoint
    print(f"Checking if Endpoint {endpoint_name} exists...")
    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except sagemaker_client.exceptions.ClientError:
        endpoint_exists = False

    if endpoint_exists:
        print(f"Updating existing Endpoint: {endpoint_name}...")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        print(f"Creating new Endpoint: {endpoint_name}...")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )

    # Optional: Wait for the endpoint to be in service
    print(f"Waiting for endpoint {endpoint_name} to be in service. This may take several minutes...")
    waiter = sagemaker_client.get_waiter('endpoint_in_service')
    try:
        waiter.wait(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} is in service!")
    except Exception as e:
        print(f"Error waiting for endpoint to be in service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    deploy()