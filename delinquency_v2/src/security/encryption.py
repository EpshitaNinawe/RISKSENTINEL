from cryptography.fernet import Fernet
import os

def generate_key():
    key = Fernet.generate_key()
    with open("models/secret.key", "wb") as f:
        f.write(key)

def load_key():
    return open("models/secret.key", "rb").read()

def encrypt_data(data):
    key = load_key()
    cipher = Fernet(key)
    return cipher.encrypt(data.encode())

def decrypt_data(token):
    key = load_key()
    cipher = Fernet(key)
    return cipher.decrypt(token).decode()

if __name__ == "__main__":
    generate_key()
    print("Encryption key generated.")
