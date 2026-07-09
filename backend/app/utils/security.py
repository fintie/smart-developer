import bcrypt


def _password_bytes(password: str) -> bytes:
    data = password.encode("utf-8")
    if len(data) > 72:
        raise ValueError("Password cannot be longer than 72 bytes")
    return data


def get_hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(_password_bytes(password), hashed_password.encode("utf-8"))
