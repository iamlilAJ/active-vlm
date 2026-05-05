"""MCP-backed vision tools.

The example configs use only cropping because the other tools require separate
service deployments. Their wrappers are kept here so downstream users can opt in
by adding the corresponding tools to their own configs.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from cv_agent.core.registries import tool_registry
from cv_agent.tools.base import McpToolBase


class MinerUOCRInput(BaseModel):
    image_url: str = Field(
        description="Complete image access URL with protocol header and full path for OCR."
    )


class MinerUOCRTool(McpToolBase):
    """Client wrapper for MinerU OCR MCP endpoints."""

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "ocr",
        description: str | None = None,
        args_schema: type[BaseModel] = MinerUOCRInput,
        remote_tool_name: str = "parse_pdf_file_parse_post",
    ) -> None:
        if description is None:
            description = "Extract text from an image using an OCR service."

        self.remote_tool_name = remote_tool_name
        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str):
        payload = {"urls": [image_url], "lang_list": ["en"]}

        async with self._client as client:
            return await client.call_tool(self.remote_tool_name, payload)


@tool_registry.register("ocr")
def get_mcp_mineru_ocr_tool(
    server_url: str,
    name: str = "ocr",
    description: str | None = None,
    args_schema: type[BaseModel] = MinerUOCRInput,
    remote_tool_name: str = "parse_pdf_file_parse_post",
) -> StructuredTool:
    return MinerUOCRTool(
        server_url,
        name,
        description,
        args_schema,
        remote_tool_name,
    ).langchain_tool


class DetectionInput(BaseModel):
    image_url: str = Field(
        description="Complete image access URL with protocol header and full path for detection."
    )
    targets: list[str] = Field(
        description=(
            "Object labels to detect. Use concise English nouns or noun phrases, "
            "for example 'red car' or 'glass bottle'."
        ),
        examples=["red car", "glass bottle"],
    )


DETECTION_DESCRIPTION = (
    "Detect objects in an image from concise natural-language labels. "
    "Rewrite complex descriptions into short modifier-noun phrases before calling."
)


class DetectionTool(McpToolBase):
    """Client wrapper for the standard detection MCP endpoint."""

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "detection",
        description: str | None = None,
        args_schema: type[BaseModel] = DetectionInput,
    ) -> None:
        super().__init__(
            mcp_server_url,
            name,
            description or DETECTION_DESCRIPTION,
            args_schema,
        )

    async def invoke(self, image_url: str, targets: list[str]):
        payload = {"image_url": image_url, "text": targets}

        async with self._client as client:
            return await client.call_tool("llmdet_detection_llmdet_detection_post", payload)


@tool_registry.register("detection")
def get_mcp_detection_tool(
    server_url: str,
    name: str = "detection",
    description: str | None = None,
    args_schema: type[BaseModel] = DetectionInput,
) -> StructuredTool:
    return DetectionTool(server_url, name, description, args_schema).langchain_tool


class DetectionCropMixTool(McpToolBase):
    """Client wrapper for the high-recall small-object detection MCP endpoint."""

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "detection_small_object",
        description: str | None = None,
        args_schema: type[BaseModel] = DetectionInput,
    ) -> None:
        default_description = (
            "High-recall detection for small objects. This may return overlapping boxes "
            f"for large objects. {DETECTION_DESCRIPTION}"
        )
        super().__init__(
            mcp_server_url,
            name,
            description or default_description,
            args_schema,
        )

    async def invoke(self, image_url: str, targets: list[str]):
        payload = {"image_url": image_url, "text": targets}

        async with self._client as client:
            return await client.call_tool(
                "llmdet_detection_crop_mix_llmdet_detection_crop_mix_post",
                payload,
            )


@tool_registry.register("detection_small_object")
def get_mcp_detection_crop_mix_tool(
    server_url: str,
    name: str = "detection_small_object",
    description: str | None = None,
    args_schema: type[BaseModel] = DetectionInput,
) -> StructuredTool:
    return DetectionCropMixTool(server_url, name, description, args_schema).langchain_tool


class SegmentationInput(BaseModel):
    image_url: str = Field(
        description="Complete image access URL with protocol header and full path."
    )
    text_prompt: str = Field(
        description="Text description of the object to segment from the image."
    )


class SegmentationTool(McpToolBase):
    """Client wrapper for the segmentation MCP endpoint."""

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "segmentation",
        description: str | None = None,
        args_schema: type[BaseModel] = SegmentationInput,
    ) -> None:
        if description is None:
            description = "Generate segmentation masks from an image and a text prompt."

        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str, text_prompt: str):
        payload = {"image_url": image_url, "text_prompt": text_prompt}

        async with self._client as client:
            return await client.call_tool("segmentation_segmentation_post", payload)


@tool_registry.register("segmentation")
def get_mcp_segmentation_tool(
    server_url: str,
    name: str = "segmentation",
    description: str | None = None,
    args_schema: type[BaseModel] = SegmentationInput,
) -> StructuredTool:
    return SegmentationTool(server_url, name, description, args_schema).langchain_tool


class CropImageInput(BaseModel):
    image_url: str = Field(
        description="Complete image access URL with protocol header and full path."
    )
    coordinates: list[float] = Field(
        description="Crop coordinates as [x1, y1, x2, y2] in absolute pixels."
    )

    x1: int = Field(exclude=True, default=0)
    y1: int = Field(exclude=True, default=0)
    x2: int = Field(exclude=True, default=0)
    y2: int = Field(exclude=True, default=0)

    @field_validator("coordinates")
    @classmethod
    def convert_coordinates(cls, value: list[float], values: ValidationInfo) -> list[float]:
        if len(value) != 4:
            raise ValueError("Coordinates list must contain exactly four values.")

        values.data["x1"] = int(value[0])
        values.data["y1"] = int(value[1])
        values.data["x2"] = int(value[2])
        values.data["y2"] = int(value[3])
        return value


class CroppingTool(McpToolBase):
    """Client wrapper for the remote crop image MCP endpoint."""

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "cropping",
        description: str | None = None,
        args_schema: type[BaseModel] = CropImageInput,
    ) -> None:
        if description is None:
            description = (
                "Precise image cropping tool. Crops an image using coordinates [x1, y1, x2, y2]."
            )

        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str, coordinates: list[float]):
        x1, y1, x2, y2 = (int(value) for value in coordinates)
        payload = {"image_url": image_url, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

        async with self._client as client:
            return await client.call_tool("crop_image_tool_crop_image_post", payload)


@tool_registry.register("cropping")
def get_mcp_cropping_tool(
    server_url: str,
    name: str = "cropping",
    description: str | None = None,
    args_schema: type[BaseModel] = CropImageInput,
) -> StructuredTool:
    return CroppingTool(server_url, name, description, args_schema).langchain_tool


class DepthEstimationInput(BaseModel):
    image_url: str = Field(
        description="Complete image access URL with protocol header and full path."
    )


class DepthEstimationTool(McpToolBase):
    """Client wrapper for the depth-estimation MCP endpoint."""

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "depth_estimation",
        description: str | None = None,
        args_schema: type[BaseModel] = DepthEstimationInput,
    ) -> None:
        if description is None:
            description = "Generate a depth map from a single input image."

        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str):
        payload = {"image_url": image_url}

        async with self._client as client:
            return await client.call_tool("depth_estimation_depth_estimation_post", payload)


@tool_registry.register("depth_estimation")
def get_mcp_depth_estimation_tool(
    server_url: str,
    name: str = "depth_estimation",
    description: str | None = None,
    args_schema: type[BaseModel] = DepthEstimationInput,
) -> StructuredTool:
    return DepthEstimationTool(server_url, name, description, args_schema).langchain_tool
