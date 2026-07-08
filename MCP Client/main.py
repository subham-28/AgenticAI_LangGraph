# calculator_server.py
from __future__ import annotations
from fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("arith")

def _as_number(x):
    # Accept ints/floats or numeric strings; raise clean errors otherwise
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError("Expected a number (int/float or numeric string)")

@mcp.tool()
async def add(a: float, b: float) -> float:
    """Return a + b."""
    return _as_number(a) + _as_number(b)

@mcp.tool()
async def subtract(a: float, b: float) -> float:
    """Return a - b."""
    return _as_number(a) - _as_number(b)

@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Return a * b."""
    return _as_number(a) * _as_number(b)

@mcp.tool()
async def divide(a: float, b: float) -> float:
    """Return a / b. Raises ValueError on division by zero."""
    num_b = _as_number(b)
    if num_b == 0.0:
        raise ValueError("Mathematical Error: Cannot divide by zero.")
    return _as_number(a) / num_b

@mcp.tool()
async def modulo(a: float, b: float) -> float:
    """Return a % b (the remainder of a / b). Raises ValueError if b is zero."""
    num_b = _as_number(b)
    if num_b == 0.0:
        raise ValueError("Mathematical Error: Cannot calculate modulo by zero.")
    return _as_number(a) % num_b

@mcp.tool()
async def power(a: float, b: float) -> float:
    """Return a ** b (a raised to the power of b)."""
    return _as_number(a) ** _as_number(b)

if __name__ == "__main__":
    # Start the server using standard standard I/O (default for FastMCP)
    mcp.run()