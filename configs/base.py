from pydantic import BaseModel, ConfigDict


class ConfigBase(BaseModel):
    """
    Common base class for all config models.

    - Provides strict validation by forbidding unknown fields.
    - Serves as the root for all configuration schemas.
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )


class DiscriminatedConfigBase(ConfigBase):
    """
    Base class for configs used in discriminated unions.

    - Requires a 'type' field for discrimination.
    - Subclasses must override 'type' with a Literal value.
    """
    type: str

    def to_dict_no_type(self):
        return self.model_dump(exclude={"type"})
