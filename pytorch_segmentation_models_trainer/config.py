try:
    from pydantic_settings import BaseSettings
except (ImportError, AttributeError):
    try:
        from pydantic import BaseSettings
    except (ImportError, AttributeError):
        # Fallback for Pydantic v2 where it might raise PydanticImportError
        # or if we want to be very safe
        try:
            from pydantic.v1 import BaseSettings
        except (ImportError, AttributeError):
            from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Awesome API"
    config_path: str
    config_name: str
