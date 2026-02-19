import hashlib
import hmac

SECRET_KEY = b'xyzxyz'

def tokenize_pii(value):
    return hmac.new(
        SECRET_KEY,
        str(value).encode(),
        hashlib.sha256
    ).hexdigest()
