import json
import requests
import argparse

def post(body_path):
    
    body = json.load(open(body_path, 'r'))
    response = requests.post('https://km-salary-predictor.herokuapp.com/predict/', json=body)

    response_body = {
        "status_code": response.status_code,
        "response": response.json()
    }
    print(json.dumps(response_body, indent=2))
    return response_body
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A post that returns response data")


    parser.add_argument(
        "--body", 
        type=str,
        help='A path to the json body to post',
        required=False,
        default='./test_body.json'
    )
    
    args = parser.parse_args()
    
    post(args.body)