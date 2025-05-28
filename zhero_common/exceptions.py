# zhero_common/exceptions.py
from typing import Any

class ZHeroException(Exception):
    """Base exception for all Z-HERO specific errors."""
    def __init__(self, message: str, status_code: int = 500, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)

class ZHeroAgentError(ZHeroException):
    """Raised when an internal agent encounters an error."""
    def __init__(self, agent_name: str, message: str, status_code: int = 500, original_error: Any = None):
        super().__init__(f"Agent {agent_name} error: {message}", status_code, original_error)
        self.agent_name = agent_name
        self.original_error = original_error

class ZHeroAuthError(ZHeroException):
    """Raised for authentication or authorization failures."""
    def __init__(self, message: str = "Authentication or authorization failed.", status_code: int = 401):
        super().__init__(message, status_code)

class ZHeroNotFoundError(ZHeroException):
    """Raised when a requested resource is not found."""
    def __init__(self, resource_name: str = "Resource", identifier: str = None, status_code: int = 404):
        msg = f"{resource_name} not found"
        if identifier:
            msg += f" with identifier '{identifier}'"
        super().__init__(msg, status_code)
        self.resource_name = resource_name
        self.identifier = identifier

class ZHeroInvalidInputError(ZHeroException):
    """Raised when input validation fails."""
    def __init__(self, message: str = "Invalid input provided.", details: Any = None, status_code: int = 400):
        super().__init__(message, status_code, details)

class ZHeroDependencyError(ZHeroAgentError):
    """Raised when an agent's external dependency fails."""
    def __init__(self, agent_name: str, dependency: str, message: str, status_code: int = 503, original_error: Any = None):
        super().__init__(
            agent_name,
            f"Dependency '{dependency}' failed: {message}",
            status_code,
            original_error
        )
        self.dependency = dependency

# --- Specific Examples ---
class ZHeroSupabaseError(ZHeroDependencyError):
    """Raised when Supabase operations fail."""
    def __init__(self, agent_name: str, message: str = "Supabase operation failed.", original_error: Any = None):
        super().__init__(agent_name, "Supabase", message, 500, original_error)

class ZHeroVertexAIError(ZHeroDependencyError):
    """Raised when Vertex AI services fail."""
    def __init__(self, agent_name: str, service: str, message: str = "Vertex AI operation failed.", original_error: Any = None):
        super().__init__(agent_name, f"Vertex AI {service}", message, 500, original_error)
