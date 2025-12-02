from mcp.server.fastmcp import FastMCP

# MCP 서버 정의 ####################################
mcp = FastMCP("Math")
@mcp.tool()
def add(a: int, b: int) -> int:
    """두 숫자 더하기"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """두 숫자 곱하기"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")