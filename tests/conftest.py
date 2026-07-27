from unittest.mock import MagicMock, patch

# memory_lab.agents.graph builds a ChromaDB HttpClient at *import* time
# (module-level `ltm = LongTermMemory()`), and chromadb.HttpClient() itself
# connects eagerly -- so this must be patched before any test module
# imports memory_lab.api, or collecting tests requires a live ChromaDB
# server. Started here (top-level, not inside a fixture) since conftest.py
# is loaded before test module imports are resolved.
_chroma_patcher = patch("chromadb.HttpClient", return_value=MagicMock())
_chroma_patcher.start()
