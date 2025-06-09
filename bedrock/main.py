from anthropic import AnthropicBedrock
import os
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()

    #import boto3

    #bedrock = boto3.client(service_name="bedrock")
    #response = bedrock.list_foundation_models(byProvider="anthropic")

    #for summary in response["modelSummaries"]:
    #    print(summary["modelId"])

    #import json

    #bedrock = boto3.client(service_name="bedrock-runtime")
    #body = json.dumps({
    #"max_tokens": 256,
    #"messages": [{"role": "user", "content": "Hello, world"}],
    #"anthropic_version": "bedrock-2023-05-31"
    #})

    #response = bedrock.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20241022-v2:0")

    #response_body = json.loads(response.get("body").read())
    #print(response_body.get("content"))

    client = AnthropicBedrock(
        # Authenticate by either providing the keys below or use the default AWS credential providers, such as
        # using ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_region="us-west-2",
    )

    message = client.messages.create(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens=256,
        messages=[{"role": "user", "content": "Hello, world"}]
    )
    print(message.content)


if __name__ == "__main__":
    main()
