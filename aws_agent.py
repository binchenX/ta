import json
import os
from typing import Dict, List

import boto3
from langchain.schema import AIMessage, FunctionMessage, HumanMessage
from langchain_openai import ChatOpenAI


class AWSAgent:
    def __init__(self):
        # Load AWS credentials from the default profile
        session = boto3.Session()
        self.credentials = session.get_credentials()
        self.model_name = "gpt-4o-mini_v2024-07-18"
        self.api_key = os.environ.get("OPENAI_API_KEY")
        # Initialize AWS S3 client with the loaded credentials
        self.s3_client = boto3.client("s3")
        # Define available functions
        self.functions = [
            {
                "name": "list_buckets",
                "description": "Lists all S3 buckets in the AWS account",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "list_objects",
                "description": "Lists objects in a specific S3 bucket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bucket_name": {"type": "string", "description": "Name of the S3 bucket"}
                    },
                    "required": ["bucket_name"],
                },
            },
        ]
        # Initialize ChatOpenAI
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_name,
            openai_api_key=self.api_key,
        )

    def list_buckets(self) -> List[str]:
        """List all S3 buckets"""
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket["Name"] for bucket in response["Buckets"]]
            return buckets
        except Exception as e:
            return f"Error listing buckets: {str(e)}"

    def list_objects(self, bucket_name: str) -> List[str]:
        """List objects in a specific bucket"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name)
            if "Contents" in response:
                objects = [obj["Key"] for obj in response["Contents"]]
                return objects
            return []
        except Exception as e:
            return f"Error listing objects: {str(e)}"

    def execute_function(self, function_name: str, function_args: Dict) -> str:
        """Execute the specified function with given arguments"""
        if function_name == "list_buckets":
            return self.list_buckets()
        elif function_name == "list_objects":
            return self.list_objects(function_args.get("bucket_name"))
        return "Function not found"

    def chat(self):
        """Main chat loop"""
        # Keep track of conversation history
        messages = []
        print(
            "S3 Assistant: Hello! I can help you with AWS S3 operations. What would you like to do?"
        )
        print("(Type 'quit' to exit)")
        while True:
            # Get user input
            user_input = input("You: ")
            if user_input.lower() == "quit":
                print("S3 Assistant: Goodbye!")
                break
            # Add user message to history
            messages.append(HumanMessage(content=user_input))
            # Get response from LLM
            response = self.llm.invoke(messages, functions=self.functions)
            # Handle function calling
            if response.additional_kwargs.get("function_call"):
                function_call = response.additional_kwargs["function_call"]
                function_name = function_call["name"]
                function_args = json.loads(function_call["arguments"])
                # Execute function
                function_response = self.execute_function(function_name, function_args)
                # Add function response to messages
                messages.append(FunctionMessage(content=str(function_response), name=function_name))
                # Get final response from LLM
                final_response = self.llm.invoke(messages)
                print("S3 Assistant:", final_response.content)
                messages.append(AIMessage(content=final_response.content))
            else:
                print("S3 Assistant:", response.content)
                messages.append(AIMessage(content=response.content))


if __name__ == "__main__":
    agent = AWSAgent()
    agent.chat()
    agent.chat()
