from langchain_core.tools import tool

class EmptyTool:
    """This is an empty tool that does nothing."""
    def __init__(self):
        pass

    @tool
    def placeholder_tool() -> str:
        """This is a placeholder tool that does nothing."""
        return "No operation performed."