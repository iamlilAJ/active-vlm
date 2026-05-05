from abc import ABC, abstractmethod
from typing import Any

import fastmcp
from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class ToolBase(ABC):
    def __init__(self, name: str, description: str, args_schema: type[BaseModel]) -> None:
        self._metadata = {"name": name, "description": description, "args_schema": args_schema}

    @abstractmethod
    async def invoke(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def langchain_tool(self) -> StructuredTool:
        return StructuredTool.from_function(coroutine=self.invoke, **self.metadata)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class McpToolBase(ToolBase):
    def __init__(self, url: str, name: str, description: str, args_schema: type[BaseModel]) -> None:
        super().__init__(name, description, args_schema)
        self._client = fastmcp.Client(url)
