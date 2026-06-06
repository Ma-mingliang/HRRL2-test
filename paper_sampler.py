"""PaperSampler: Iterative paper/method selection from the reward paper pool."""

import json
from pathlib import Path
from typing import Optional


class PaperSampler:
    """Manages iterative paper/method selection from method_pool.jsonl."""

    def __init__(self, pool_dir: Path, work_dir: Path):
        self.pool_dir = pool_dir
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Load method pool
        self.methods = []
        method_pool_path = pool_dir / "method_pool.jsonl"
        with open(method_pool_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.methods.append(json.loads(line))

        # Load taxonomy for priority ordering
        self.taxonomy = {}
        taxonomy_path = pool_dir / "taxonomy.yaml"
        if taxonomy_path.exists():
            import yaml
            with open(taxonomy_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.taxonomy = data.get("categories", {})

        # Load paper index for MD file lookup
        self.paper_dir = pool_dir / "papers" / "all_sources" / "by_category"
        self._paper_index = self._build_paper_index()

        # Load used methods tracking
        self.used_path = work_dir / "used_methods.jsonl"
        self.used_ids = self._load_used()

    def _build_paper_index(self) -> dict:
        """Build index mapping openreview IDs to MD file paths."""
        index = {}
        if not self.paper_dir.exists():
            return index
        for md_file in self.paper_dir.rglob("*.md"):
            name = md_file.stem
            # Extract openreview ID from filename like "arxiv - openreview_lxFL2g3YxB - ..."
            if "openreview_" in name:
                parts = name.split("openreview_")
                if len(parts) > 1:
                    oid = parts[1].split(" - ")[0].split(" ")[0]
                    index[oid.lower()] = md_file
            # Extract arxiv ID
            if "arxiv_" in name:
                parts = name.split("arxiv_")
                if len(parts) > 1:
                    aid = parts[1].split(" - ")[0].split(" ")[0]
                    index[f"arxiv:{aid}"] = md_file
        return index

    def _load_used(self) -> set:
        """Load set of already-used method IDs."""
        if not self.used_path.exists():
            return set()
        used = set()
        with open(self.used_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    used.add(data.get("method_id", ""))
        return used

    def _save_used(self, method_id: str, category: str, result: str):
        """Append a used method record."""
        record = {
            "method_id": method_id,
            "category": category,
            "result": result,
        }
        with open(self.used_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def mark_used(self, method_id: str, category: str, result: str = "tried"):
        """Mark a method as used."""
        self.used_ids.add(method_id)
        self._save_used(method_id, category, result)

    def get_methods_by_category(self, category: str, skip_used: bool = True) -> list:
        """Get all methods in a category, optionally skipping used ones."""
        candidates = [m for m in self.methods if m["category"] == category]
        if skip_used:
            candidates = [m for m in candidates if m["method_id"] not in self.used_ids]
        # Sort by confidence (high > medium > low)
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        candidates.sort(key=lambda m: confidence_order.get(m.get("confidence", "low"), 2))
        return candidates

    def get_next_batch(self, batch_size: int = 1, preferred_category: Optional[str] = None) -> list:
        """Pick next methods from highest-priority untried category.

        Args:
            batch_size: Number of methods to return.
            preferred_category: If set,优先选择该类别.

        Returns:
            List of method dicts with source paper MD content attached.
        """
        # Priority order: S > A > B
        priority_map = {"S": 0, "A": 1, "B": 2}
        categories = []
        for cat_name, cat_info in self.taxonomy.items():
            priority = priority_map.get(cat_info.get("priority", "B"), 2)
            categories.append((priority, cat_name))
        categories.sort()

        # Try preferred category first
        if preferred_category:
            candidates = self.get_methods_by_category(preferred_category)
            if candidates:
                batch = candidates[:batch_size]
                for m in batch:
                    m["_paper_md"] = self._load_paper_md(m)
                return batch

        # Iterate by priority
        for _, cat_name in categories:
            candidates = self.get_methods_by_category(cat_name)
            if candidates:
                batch = candidates[:batch_size]
                for m in batch:
                    m["_paper_md"] = self._load_paper_md(m)
                return batch

        # All methods used - reset and try again
        self.used_ids.clear()
        if self.used_path.exists():
            self.used_path.unlink()
        return self.get_next_batch(batch_size, preferred_category)

    def _load_paper_md(self, method: dict) -> Optional[str]:
        """Load paper MD content for a method's source papers."""
        source_papers = method.get("source_papers", [])
        for paper_ref in source_papers:
            # Parse openreview:ID or arxiv:ID
            if ":" in paper_ref:
                prefix, pid = paper_ref.split(":", 1)
                if prefix == "openreview":
                    # Try to find in index
                    md_path = self._paper_index.get(pid.lower())
                    if md_path and md_path.exists():
                        try:
                            content = md_path.read_text(encoding="utf-8")
                            # Truncate to ~8000 chars for LLM context
                            if len(content) > 8000:
                                content = content[:8000] + "\n\n[... truncated ...]"
                            return content
                        except Exception:
                            pass
                elif prefix == "arxiv":
                    md_path = self._paper_index.get(f"arxiv:{pid}")
                    if md_path and md_path.exists():
                        try:
                            content = md_path.read_text(encoding="utf-8")
                            if len(content) > 8000:
                                content = content[:8000] + "\n\n[... truncated ...]"
                            return content
                        except Exception:
                            pass
        return None

    def remaining_categories(self) -> list:
        """List categories with untried methods."""
        result = []
        for cat_name in self.taxonomy:
            candidates = self.get_methods_by_category(cat_name)
            if candidates:
                result.append(cat_name)
        return result

    def summary(self) -> dict:
        """Return summary of used/remaining methods."""
        total = len(self.methods)
        used = len(self.used_ids)
        by_category = {}
        for cat_name in self.taxonomy:
            cat_methods = [m for m in self.methods if m["category"] == cat_name]
            cat_used = [m for m in cat_methods if m["method_id"] in self.used_ids]
            by_category[cat_name] = {
                "total": len(cat_methods),
                "used": len(cat_used),
                "remaining": len(cat_methods) - len(cat_used),
            }
        return {
            "total": total,
            "used": used,
            "remaining": total - used,
            "by_category": by_category,
        }
