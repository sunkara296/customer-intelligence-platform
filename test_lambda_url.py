import json
import subprocess
import os
import urllib.request

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

profile = os.getenv("AWS_PROFILE", "customer-intelligence-dev")
region = os.getenv("AWS_REGION", "us-east-1")
base_url = os.environ["FUNCTION_URL"].rstrip("/")

result = subprocess.check_output([
    "aws", "configure", "export-credentials",
    "--profile", profile,
    "--format", "process"
], text=True)

data = json.loads(result)

credentials = Credentials(
    data["AccessKeyId"],
    data["SecretAccessKey"],
    data.get("SessionToken")
)

for path in ["/health", "/predict/1001"]:
    url = base_url + path

    request = AWSRequest(method="GET", url=url)
    SigV4Auth(credentials, "lambda", region).add_auth(request)

    signed_request = urllib.request.Request(
        url,
        headers=dict(request.headers),
        method="GET"
    )

    with urllib.request.urlopen(signed_request, timeout=60) as response:
        print(f"\nEndpoint: {path}")
        print(f"HTTP Status: {response.status}")
        print(json.dumps(json.loads(response.read()), indent=2))
