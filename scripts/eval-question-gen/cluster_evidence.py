#!/usr/bin/env python3
"""Seen-evidence cluster construction for Eval-Question-Gen."""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source_dataset import SourceRow, doc_key_for_doc_ids, normalize_question


@dataclass
class UnionFind:
    parent: dict[str, str]
    rank: dict[str, int]

    @classmethod
    def create(cls) -> "UnionFind":
        return cls(parent={}, rank={})

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        left_rank = self.rank[left_root]
        right_rank = self.rank[right_root]
        if left_rank < right_rank:
            self.parent[left_root] = right_root
        elif left_rank > right_rank:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1

    def components(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for value in self.parent:
            result[self.find(value)].append(value)
        return {root: sorted(values) for root, values in result.items()}


def chunk_sort_key(chunk_id: str) -> tuple[str, int, str]:
    if "#" not in chunk_id:
        return (chunk_id, -1, chunk_id)
    doc_id, index_text = chunk_id.rsplit("#", 1)
    try:
        index = int(index_text)
    except ValueError:
        index = -1
    return (doc_id, index, chunk_id)


def chunk_doc_id(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0] if "#" in chunk_id else chunk_id


def chunk_index(chunk_id: str) -> int:
    if "#" not in chunk_id:
        return -1
    try:
        return int(chunk_id.rsplit("#", 1)[1])
    except ValueError:
        return -1


def row_train_keys(rows: list[SourceRow]) -> list[str]:
    return sorted({row.train_key for row in rows if row.train_key})


def source_pipelines(rows: list[SourceRow]) -> list[str]:
    return sorted({row.pipeline for row in rows if row.pipeline})


def unique_question_count(rows: list[SourceRow]) -> int:
    return len({normalize_question(row.question) for row in rows if row.question})


def rows_touching_chunks(rows: list[SourceRow], chunk_ids: set[str]) -> list[SourceRow]:
    return [
        row
        for row in rows
        if set(row.chunk_ids) & chunk_ids
    ]


def edge_density(chunk_ids: list[str], edge_counter: Counter[tuple[str, str]]) -> float:
    if len(chunk_ids) < 2:
        return 0.0
    possible_edges = len(chunk_ids) * (len(chunk_ids) - 1) / 2
    present_edges = 0
    for left, right in itertools.combinations(sorted(chunk_ids), 2):
        if edge_counter.get((left, right), 0) > 0:
            present_edges += 1
    return round(present_edges / possible_edges, 4)


def make_cluster(
    *,
    cluster_id: str,
    cluster_kind: str,
    chunk_ids: list[str],
    rows: list[SourceRow],
    edge_counter: Counter[tuple[str, str]] | None = None,
    split_from: str | None = None,
) -> dict[str, Any]:
    sorted_chunks = sorted(set(chunk_ids), key=chunk_sort_key)
    doc_ids = sorted({chunk_doc_id(chunk_id) for chunk_id in sorted_chunks})
    seed_train_keys = row_train_keys(rows)
    return {
        "cluster_id": cluster_id,
        "cluster_kind": cluster_kind,
        "chunk_ids": sorted_chunks,
        "doc_ids": doc_ids,
        "doc_key": doc_key_for_doc_ids(doc_ids),
        "seed_train_keys": seed_train_keys,
        "source_pipelines": source_pipelines(rows),
        "split_from": split_from,
        "support": {
            "row_count": len(seed_train_keys),
            "unique_question_count": unique_question_count(rows),
            "chunk_count": len(sorted_chunks),
            "doc_count": len(doc_ids),
            "edge_density": edge_density(sorted_chunks, edge_counter or Counter()),
        },
    }


def cluster_id(prefix: str, counter: int) -> str:
    return f"eqg_{prefix}_{counter:06d}"


def split_chunks_for_prompt(
    chunk_ids: list[str],
    *,
    max_chunks: int,
) -> list[list[str]]:
    sorted_chunks = sorted(set(chunk_ids), key=chunk_sort_key)
    if len(sorted_chunks) <= max_chunks:
        return [sorted_chunks]
    by_doc: dict[str, list[str]] = defaultdict(list)
    for chunk_id in sorted_chunks:
        by_doc[chunk_doc_id(chunk_id)].append(chunk_id)

    splits: list[list[str]] = []
    for doc_id in sorted(by_doc):
        doc_chunks = by_doc[doc_id]
        for start in range(0, len(doc_chunks), max_chunks):
            splits.append(doc_chunks[start : start + max_chunks])
    return splits


def build_exact_evidence_clusters(
    rows: list[SourceRow],
    *,
    start_index: int,
    edge_counter: Counter[tuple[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[tuple[str, ...], list[SourceRow]] = defaultdict(list)
    for row in rows:
        if row.evidence_signature:
            grouped[row.evidence_signature].append(row)

    clusters: list[dict[str, Any]] = []
    counter = start_index
    for signature, group_rows in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        counter += 1
        clusters.append(
            make_cluster(
                cluster_id=cluster_id("exact", counter),
                cluster_kind="exact_evidence_set",
                chunk_ids=list(signature),
                rows=group_rows,
                edge_counter=edge_counter,
            )
        )
    return clusters, counter


def build_doc_set_clusters(
    rows: list[SourceRow],
    *,
    start_index: int,
    max_chunks: int,
    edge_counter: Counter[tuple[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[tuple[str, ...], list[SourceRow]] = defaultdict(list)
    for row in rows:
        if row.doc_set_signature:
            grouped[row.doc_set_signature].append(row)

    clusters: list[dict[str, Any]] = []
    counter = start_index
    for _, group_rows in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        chunks = sorted(
            {chunk_id for row in group_rows for chunk_id in row.chunk_ids},
            key=chunk_sort_key,
        )
        split_from = None
        chunk_splits = split_chunks_for_prompt(chunks, max_chunks=max_chunks)
        if len(chunk_splits) > 1:
            split_from = "large_doc_set"
        for split in chunk_splits:
            split_rows = rows_touching_chunks(group_rows, set(split))
            if len(split) < 2:
                continue
            counter += 1
            clusters.append(
                make_cluster(
                    cluster_id=cluster_id("docset", counter),
                    cluster_kind="doc_set",
                    chunk_ids=split,
                    rows=split_rows,
                    edge_counter=edge_counter,
                    split_from=split_from,
                )
            )
    return clusters, counter


def build_co_citation_clusters(
    rows: list[SourceRow],
    *,
    start_index: int,
    max_chunks: int,
    edge_counter: Counter[tuple[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    union_find = UnionFind.create()
    for row in rows:
        unique_chunks = sorted(set(row.chunk_ids), key=chunk_sort_key)
        for chunk_id in unique_chunks:
            union_find.add(chunk_id)
        if len(unique_chunks) < 2:
            continue
        first = unique_chunks[0]
        for chunk_id in unique_chunks[1:]:
            union_find.union(first, chunk_id)

    clusters: list[dict[str, Any]] = []
    counter = start_index
    for _, component_chunks in sorted(
        union_find.components().items(), key=lambda item: (-len(item[1]), item[1])
    ):
        if len(component_chunks) < 2:
            continue
        component_rows = rows_touching_chunks(rows, set(component_chunks))
        split_from = None
        chunk_splits = split_chunks_for_prompt(component_chunks, max_chunks=max_chunks)
        if len(chunk_splits) > 1:
            split_from = "large_co_citation_component"
        for split in chunk_splits:
            split_rows = rows_touching_chunks(component_rows, set(split))
            counter += 1
            clusters.append(
                make_cluster(
                    cluster_id=cluster_id("cocite", counter),
                    cluster_kind="co_citation",
                    chunk_ids=split,
                    rows=split_rows,
                    edge_counter=edge_counter,
                    split_from=split_from,
                )
            )
    return clusters, counter


def build_doc_local_clusters(
    rows: list[SourceRow],
    *,
    start_index: int,
    max_gap: int,
    max_chunks: int,
    edge_counter: Counter[tuple[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    doc_to_chunks: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for chunk_id in row.chunk_ids:
            doc_to_chunks[chunk_doc_id(chunk_id)].add(chunk_id)

    clusters: list[dict[str, Any]] = []
    counter = start_index
    for doc_id in sorted(doc_to_chunks):
        sorted_chunks = sorted(doc_to_chunks[doc_id], key=chunk_sort_key)
        groups: list[list[str]] = []
        current: list[str] = []
        previous_index: int | None = None
        for chunk_id in sorted_chunks:
            index = chunk_index(chunk_id)
            if (
                current
                and previous_index is not None
                and index >= 0
                and index - previous_index > max_gap
            ):
                groups.append(current)
                current = []
            current.append(chunk_id)
            previous_index = index
        if current:
            groups.append(current)

        for group in groups:
            if len(group) < 2:
                continue
            for split in split_chunks_for_prompt(group, max_chunks=max_chunks):
                if len(split) < 2:
                    continue
                counter += 1
                clusters.append(
                    make_cluster(
                        cluster_id=cluster_id("doclocal", counter),
                        cluster_kind="doc_local_seen_chunks",
                        chunk_ids=split,
                        rows=rows_touching_chunks(rows, set(split)),
                        edge_counter=edge_counter,
                        split_from="doc_local_large_group"
                        if len(group) > len(split)
                        else None,
                    )
                )
    return clusters, counter


def build_multi_doc_bridge_clusters(
    rows: list[SourceRow],
    *,
    start_index: int,
    max_chunks: int,
    edge_counter: Counter[tuple[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[tuple[str, ...], list[SourceRow]] = defaultdict(list)
    for row in rows:
        if len(row.doc_set_signature) > 1:
            grouped[row.doc_set_signature].append(row)

    clusters: list[dict[str, Any]] = []
    counter = start_index
    for _, group_rows in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        chunks = sorted(
            {chunk_id for row in group_rows for chunk_id in row.chunk_ids},
            key=chunk_sort_key,
        )
        split_from = None
        chunk_splits = split_chunks_for_prompt(chunks, max_chunks=max_chunks)
        if len(chunk_splits) > 1:
            split_from = "large_multi_doc_bridge"
        for split in chunk_splits:
            split_rows = rows_touching_chunks(group_rows, set(split))
            if len({chunk_doc_id(chunk_id) for chunk_id in split}) < 2:
                continue
            counter += 1
            clusters.append(
                make_cluster(
                    cluster_id=cluster_id("multidoc", counter),
                    cluster_kind="multi_doc_bridge",
                    chunk_ids=split,
                    rows=split_rows,
                    edge_counter=edge_counter,
                    split_from=split_from,
                )
            )
    return clusters, counter


def build_edge_counter(rows: list[SourceRow]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        unique_chunks = sorted(set(row.chunk_ids), key=chunk_sort_key)
        for left, right in itertools.combinations(unique_chunks, 2):
            counter[(left, right)] += 1
    return counter


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_seen_evidence_clusters(
    rows: list[SourceRow],
    *,
    run_dir: Path,
    max_chunks_per_cluster: int = 30,
    doc_local_max_gap: int = 3,
    limit_clusters: int | None = None,
    focus_train_keys: set[str] | None = None,
) -> dict[str, Any]:
    all_rows = rows
    generation_rows = [row for row in rows if not row.is_exact_duplicate]

    edge_counter = build_edge_counter(generation_rows)
    clusters: list[dict[str, Any]] = []
    next_counter = 0

    exact_clusters, next_counter = build_exact_evidence_clusters(
        generation_rows,
        start_index=next_counter,
        edge_counter=edge_counter,
    )
    clusters.extend(exact_clusters)

    doc_set_clusters, next_counter = build_doc_set_clusters(
        generation_rows,
        start_index=next_counter,
        max_chunks=max_chunks_per_cluster,
        edge_counter=edge_counter,
    )
    clusters.extend(doc_set_clusters)

    cocite_clusters, next_counter = build_co_citation_clusters(
        generation_rows,
        start_index=next_counter,
        max_chunks=max_chunks_per_cluster,
        edge_counter=edge_counter,
    )
    clusters.extend(cocite_clusters)

    doc_local_clusters, next_counter = build_doc_local_clusters(
        generation_rows,
        start_index=next_counter,
        max_gap=doc_local_max_gap,
        max_chunks=max_chunks_per_cluster,
        edge_counter=edge_counter,
    )
    clusters.extend(doc_local_clusters)

    multi_doc_clusters, next_counter = build_multi_doc_bridge_clusters(
        generation_rows,
        start_index=next_counter,
        max_chunks=max_chunks_per_cluster,
        edge_counter=edge_counter,
    )
    clusters.extend(multi_doc_clusters)

    clusters_before_focus = len(clusters)
    if focus_train_keys is not None:
        focus = set(focus_train_keys)
        clusters = [
            cluster
            for cluster in clusters
            if set(str(key) for key in cluster.get("seed_train_keys", [])) & focus
        ]

    clusters.sort(
        key=lambda cluster: (
            cluster["cluster_kind"],
            -cluster["support"]["row_count"],
            -cluster["support"]["chunk_count"],
            cluster["cluster_id"],
        )
    )
    total_clusters_before_limit = len(clusters)
    if limit_clusters is not None:
        clusters = clusters[: max(0, limit_clusters)]

    clusters_path = run_dir / "clusters.jsonl"
    summary_path = run_dir / "cluster_summary.json"
    write_jsonl(clusters, clusters_path)

    kind_counter = Counter(cluster["cluster_kind"] for cluster in clusters)
    split_counter = Counter(
        cluster["split_from"] or "<not_split>" for cluster in clusters
    )
    chunk_counts = [cluster["support"]["chunk_count"] for cluster in clusters]
    doc_counts = [cluster["support"]["doc_count"] for cluster in clusters]
    row_counts = [cluster["support"]["row_count"] for cluster in clusters]

    summary = {
        "phase": "cluster",
        "clustering_mode": "metadata_only",
        "all_source_row_count": len(all_rows),
        "generation_source_row_count": len(generation_rows),
        "max_chunks_per_cluster": max_chunks_per_cluster,
        "doc_local_max_gap": doc_local_max_gap,
        "cluster_count": len(clusters),
        "cluster_count_before_limit": total_clusters_before_limit,
        "cluster_count_before_focus": clusters_before_focus,
        "focus_train_key_count": len(focus_train_keys) if focus_train_keys is not None else None,
        "cluster_kind_counts": dict(kind_counter),
        "split_counts": dict(split_counter),
        "distributions": {
            "chunks_per_cluster": summarize_numbers(chunk_counts),
            "docs_per_cluster": summarize_numbers(doc_counts),
            "rows_per_cluster": summarize_numbers(row_counts),
        },
        "artifacts": {
            "clusters": str(clusters_path),
            "cluster_summary": str(summary_path),
        },
        "sample_clusters": clusters[:10],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def summarize_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        median = sorted_values[mid]
    else:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return {
        "min": sorted_values[0],
        "median": round(median, 2),
        "mean": round(sum(sorted_values) / len(sorted_values), 2),
        "max": sorted_values[-1],
    }
