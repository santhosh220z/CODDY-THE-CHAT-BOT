from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Single declarative base for every ORM model."""