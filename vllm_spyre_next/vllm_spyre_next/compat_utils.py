from dataclasses import fields


def dataclass_fields(cls: type) -> list[str]:
    return [f.name for f in fields(cls)]
