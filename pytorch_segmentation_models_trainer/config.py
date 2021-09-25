from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Awesome API"
    config_path: str
    config_name: str
