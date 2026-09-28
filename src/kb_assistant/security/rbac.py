"""Role-based access control.

Two things are controlled, and both are enforced in code, never in a prompt:

1. Which tools a role may execute (`Permission`). The tool executor checks this on every call,
   so a model that invents a tool name or is talked into calling an admin tool is refused.
2. Which documents a role may read (`access_level` metadata). The retriever ANDs this filter into
   every vector query; the model can narrow a search but can never widen it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


class Permission(StrEnum):
    CHAT = "chat"
    SEARCH = "search"
    ANALYTICS = "analytics"
    MCP_READ = "mcp_read"
    MCP_WRITE = "mcp_write"
    ADMIN = "admin"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset({Permission.CHAT, Permission.SEARCH}),
    Role.ANALYST: frozenset(
        {Permission.CHAT, Permission.SEARCH, Permission.ANALYTICS, Permission.MCP_READ}
    ),
    Role.ADMIN: frozenset(Permission),
}

# Ordered from least to most sensitive (see data/corpus/policy: Data Classification Policy).
ACCESS_LEVELS: tuple[str, ...] = ("public", "internal", "confidential", "restricted")

ROLE_ACCESS_LEVELS: dict[Role, frozenset[str]] = {
    Role.VIEWER: frozenset({"public", "internal"}),
    Role.ANALYST: frozenset({"public", "internal", "confidential"}),
    Role.ADMIN: frozenset(ACCESS_LEVELS),
}


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. Built from a verified JWT, passed to the graph as runtime
    context (not state), so nothing the model writes can change who the caller is."""

    user_id: str
    name: str
    role: Role
    department: str
    permissions: frozenset[Permission] = field(default_factory=frozenset)

    @classmethod
    def for_role(cls, user_id: str, name: str, role: Role | str, department: str) -> Principal:
        role = Role(role)
        return cls(user_id, name, role, department, ROLE_PERMISSIONS[role])

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions

    @property
    def access_levels(self) -> frozenset[str]:
        return ROLE_ACCESS_LEVELS[self.role]

    def can_read(self, access_level: str) -> bool:
        return access_level in self.access_levels
