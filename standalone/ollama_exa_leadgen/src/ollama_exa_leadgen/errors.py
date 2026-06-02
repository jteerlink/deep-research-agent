"""Explicit exception types for the standalone leadgen tool."""


class LeadgenError(Exception):
    """Base class for user-facing leadgen errors."""


class ConfigError(LeadgenError):
    """Configuration or missing-secret problem."""


class ValidationError(LeadgenError):
    """Invalid input or schema problem."""


class ProviderError(LeadgenError):
    """Remote provider request failed."""
