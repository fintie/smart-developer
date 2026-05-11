from __future__ import annotations
from algorithm.src.mlops.db import engine
from algorithm.src.mlops.models import Base


def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("MLOps database tables created successfully.")


if __name__ == "__main__":
    main()