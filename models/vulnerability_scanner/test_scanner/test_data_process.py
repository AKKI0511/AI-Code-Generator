import os
import json
import subprocess
from typing import Dict, Any, List
import requests

class DataProcessor:
    def __init__(self):
        # Vulnerability: Hardcoded credentials
        self.api_key = "1234567890abcdef"
        self.api_secret = "super_secret_token_123"
        self.api_endpoint = "https://api.example.com/data"
    
    def process_user_input(self, user_data: str) -> Dict[str, Any]:
        # Vulnerability: Unsafe deserialization
        try:
            return json.loads(user_data)
        except json.JSONDecodeError as e:
            print(f"Error processing input: {str(e)}")
            return {}
    
    def execute_command(self, command: str) -> str:
        # Vulnerability: Command injection
        try:
            # Vulnerability: Using shell=True
            result = subprocess.check_output(command, shell=True)
            return result.decode()
        except subprocess.CalledProcessError as e:
            print(f"Command execution error: {str(e)}")
            return ""
    
    def fetch_external_data(self, url: str) -> Dict[str, Any]:
        # Vulnerability: SSRF
        try:
            response = requests.get(url, verify=False)  # Vulnerability: SSL verification disabled
            return response.json()
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return {}
    
    def save_user_file(self, filename: str, content: bytes) -> bool:
        # Vulnerability: Path traversal
        try:
            with open(f"user_files/{filename}", "wb") as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return False
    
    def process_batch_data(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Vulnerability: No input validation
        processed_data = []
        for item in data_list:
            # Vulnerability: Potential XSS if output is rendered in web context
            if "name" in item:
                item["processed_name"] = f"<script>alert('{item['name']}')</script>"
            processed_data.append(item)
        return processed_data
    
    def export_data(self, data: Dict[str, Any], export_path: str) -> bool:
        # Vulnerability: Insecure file permissions
        try:
            with open(export_path, 'w') as f:
                json.dump(data, f)
            os.chmod(export_path, 0o777)  # Vulnerability: Overly permissive file permissions
            return True
        except Exception as e:
            print(f"Export error: {str(e)}")
            return False