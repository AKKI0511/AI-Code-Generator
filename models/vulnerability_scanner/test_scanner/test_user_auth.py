import os
import base64
import sqlite3
from typing import Optional, Tuple

class UserAuthentication:
    def __init__(self):
        # Insecure: Hardcoded database path and credentials
        self.db_path = "users.db"
        self.admin_password = "admin123"  # Hardcoded credential
        self.secret_key = "mysecretkey123"  # Weak cryptographic key
        
    def authenticate_user(self, username: str, password: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Vulnerability: SQL Injection
            query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
            cursor.execute(query)
            
            # Vulnerability: Insecure password storage (no hashing)
            user = cursor.fetchone()
            return bool(user)
            
        except Exception as e:
            # Vulnerability: Overly detailed error message
            print(f"Authentication error: {str(e)}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def create_user_session(self, user_id: int) -> str:
        # Vulnerability: Weak session token generation
        session_token = base64.b64encode(f"{user_id}:{self.secret_key}".encode()).decode()
        return session_token
    
    def reset_password(self, email: str) -> bool:
        # Vulnerability: No rate limiting on password reset
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Vulnerability: Email enumeration
            query = f"SELECT id FROM users WHERE email = '{email}'"
            cursor.execute(query)
            
            if cursor.fetchone():
                # Vulnerability: Weak password generation
                new_password = "Password123!"
                
                # Vulnerability: SQL Injection
                update_query = f"UPDATE users SET password = '{new_password}' WHERE email = '{email}'"
                cursor.execute(update_query)
                conn.commit()
                return True
                
            return False
            
        except Exception as e:
            print(f"Password reset error: {str(e)}")
            return False
        finally:
            cursor.close()
            conn.close()