"""Global symbol table to disambiguate notation across sections."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Symbol(BaseModel):
    """A mathematical / typographic symbol with its current binding."""

    name: str = Field(description="Symbol as written, e.g. 'X', '\\mathcal{H}', '\\theta'.")
    description: str = Field(description="What the symbol stands for in the current scope.")
    introduced_in_block: str
    scope_blocks: list[str] = Field(
        default_factory=list,
        description="Blocks where this binding is in scope. Empty means global.",
    )


class SymbolTable(BaseModel):
    """Tracks symbol meanings; sub-agents must update on block transitions.

    Same glyph (e.g. 'X') can mean different things in different sections.
    Without this, B group can't reliably challenge consistency, and A group
    can drift silently.
    """

    bindings: dict[str, list[Symbol]] = Field(default_factory=dict)

    def add(self, symbol: Symbol) -> None:
        self.bindings.setdefault(symbol.name, []).append(symbol)

    def lookup(self, name: str, block_id: str) -> Symbol | None:
        candidates = self.bindings.get(name, [])
        # Prefer most-recent binding scoped to block_id, fall back to global.
        for sym in reversed(candidates):
            if not sym.scope_blocks or block_id in sym.scope_blocks:
                return sym
        return candidates[-1] if candidates else None

    def all_symbols(self) -> list[Symbol]:
        return [s for syms in self.bindings.values() for s in syms]
