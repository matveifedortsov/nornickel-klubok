"""RBAC: API-key -> роль, простая проверка доступа (§7 плана).

Без внешнего IdP — для объёма хакатона это оверинжиниринг. Роль резолвится
по заголовку X-API-Key через config.settings.api_keys (переопределяется в
.env как JSON). Без ключа — самая ограниченная роль (external_partner).
"""
from __future__ import annotations

from fastapi import Header, HTTPException, Depends

from config import settings

# researcher/analyst/project_lead/admin видят внутренние данные и аналитику;
# external_partner — нет (см. sensitivity на Publication, klubok/ontology.py).
ROLES_WITH_FULL_ACCESS = {"researcher", "analyst", "project_lead", "admin"}
ROLES_THAT_CAN_EDIT_GRAPH = {"project_lead", "admin"}


def get_role(x_api_key: str | None = Header(default=None)) -> str:
    if x_api_key is None:
        return "external_partner"
    role = settings.api_keys.get(x_api_key)
    if role is None:
        raise HTTPException(status_code=401, detail="Неизвестный API-key")
    return role


def require_full_access(role: str = Depends(get_role)) -> str:
    if role not in ROLES_WITH_FULL_ACCESS:
        raise HTTPException(status_code=403, detail=f"Роль '{role}' не имеет доступа к этому ресурсу")
    return role


def require_editor(role: str = Depends(get_role)) -> str:
    if role not in ROLES_THAT_CAN_EDIT_GRAPH:
        raise HTTPException(status_code=403, detail=f"Роль '{role}' не может редактировать граф")
    return role
