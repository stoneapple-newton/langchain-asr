from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel


def structured_output_chain(llm: Any, prompt: Any, schema: type[BaseModel]):
    """Use native structured output when available, with JSON parsing fallback."""
    parser = JsonOutputParser()
    try:
        structured_llm = llm.with_structured_output(schema)
    except (AttributeError, NotImplementedError, ValueError):
        return prompt | llm | parser
    return prompt | structured_llm | _to_dict


def _to_dict(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value
