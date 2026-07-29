import boto3
import json

def test_sagemaker_endpoint():
    # Initialize the SageMaker runtime client
    client = boto3.client('sagemaker-runtime')
    
    # This must match the ENDPOINT_NAME in pipeline.yaml
    endpoint_name = 'practice-firstmodel-endpoint'

    print(f"Invoking endpoint: {endpoint_name}...")
    
    # We must provide the features in the exact same format as the training data 
    # AFTER pd.get_dummies() was applied.
    # Training columns after encoding are typically: 
    # 'age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest'
    
    payload = {
        "features": [
            {
                "age": 22,
                "bmi": 18.0,
                "children": 0,
                "sex_male": 1,
                "smoker_yes": 0,
                "region_northwest": 0,
                "region_southeast": 0,
                "region_southwest": 1
            },
            {
                "age": 35,
                "bmi": 27.9,
                "children": 2,
                "sex_male": 0,
                "smoker_yes": 1,
                "region_northwest": 1,
                "region_southeast": 0,
                "region_southwest": 0
            }
        ]
    }

    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse the JSON response
        result = json.loads(response['Body'].read().decode())
        print(f"Predictions:\n{json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"Error invoking endpoint: {e}")

if __name__ == "__main__":
    test_sagemaker_endpoint()
