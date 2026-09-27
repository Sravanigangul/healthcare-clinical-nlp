"""Drug-drug interaction provider interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class DDIResult:
    """Structured result returned by a DDI provider"""

    drug1: str
    drug2: str
    interaction_found: bool
    description: Optional[str] = None
    severity: Optional[str] = None
    evidence_score: Optional[str] = None


@dataclass
class DDIProviderResponse:
    """Complete response from a DDI provider"""

    status: str
    interactions: list[DDIResult]
    provider: str
    message: Optional[str] = None

class DDIProvider(ABC):
    """Base interface for drug-drug interaction providers."""

    @abstractmethod
    def check_interactions(
        self,
        medications: list[dict],
    ) -> DDIProviderResponse:
        """Check medications for drug-drug interactions."""
        raise NotImplementedError