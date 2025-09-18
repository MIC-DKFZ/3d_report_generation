from pydantic import BaseModel, ConfigDict


class ConfigBase(BaseModel):
    """Base class for all configs. We enforce strict validation."""
    model_config = ConfigDict(extra="forbid")
