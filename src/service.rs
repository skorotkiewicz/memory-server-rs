//! Business logic: dual-write (Neo4j first, then Qdrant), hydration through
//! the graph. Qdrant is a rebuildable index — reindex() heals any drift.
//!
//! Mirrors packages/memory-server/src/service.ts.

use crate::embeddings::EmbeddingProvider;
use crate::graph::{GraphStore, MemoryRecord, MemoryWithRelated};
use crate::vectors::VectorRecord;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub memory: MemoryRecord,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextResult {
    pub memory: MemoryRecord,
    pub score: f32,
    pub related: Vec<crate::graph::RelatedMemory>,
}

#[derive(Clone)]
pub struct MemoryService {
    graph: std::sync::Arc<dyn GraphStore>,
    vectors: std::sync::Arc<dyn crate::vectors::VectorStore>,
    embeddings: std::sync::Arc<dyn EmbeddingProvider>,
}

impl MemoryService {
    pub fn new(
        graph: std::sync::Arc<dyn GraphStore>,
        vectors: std::sync::Arc<dyn crate::vectors::VectorStore>,
        embeddings: std::sync::Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            graph,
            vectors,
            embeddings,
        }
    }

    pub async fn store(&self, text: &str, tags: Option<Vec<String>>) -> anyhow::Result<MemoryRecord> {
        // 1. graph first (source of truth)
        let record = self.graph.store_memory(text, tags).await?;
        // 2. then vector index (rebuildable)
        let result = self.embeddings.embed(text).await?;
        self.vectors
            .upsert(&VectorRecord {
                memory_id: record.id.clone(),
                vector: result.vector,
                model: result.model,
            })
            .await?;
        Ok(record)
    }

    pub async fn link(&self, from_id: &str, to_id: &str, relation: &str) -> anyhow::Result<crate::graph::LinkResult> {
        self.graph.link_memories(from_id, to_id, relation).await
    }

    pub async fn get(&self, id: &str) -> anyhow::Result<Option<MemoryWithRelated>> {
        self.graph.get_memory(id).await
    }

    pub async fn delete(&self, id: &str) -> anyhow::Result<bool> {
        let deleted = self.graph.delete_memory(id).await?;
        if deleted {
            // orphaned vectors are harmless (hydration goes through Neo4j),
            // but clean them anyway; index deletions must never fail the call
            let _ = self.vectors.delete(id).await;
        }
        Ok(deleted)
    }

    /// Semantic search: embed query → Qdrant top-k → hydrate from Neo4j.
    /// Skips orphaned vectors (missing in the graph) instead of failing.
    pub async fn search(&self, query: &str, top_k: Option<usize>) -> anyhow::Result<Vec<SearchResult>> {
        let top_k = top_k.unwrap_or(5);
        let result = self.embeddings.embed(query).await?;
        let hits = self.vectors.search(&result.vector, top_k).await?;
        let mut results = Vec::new();
        for hit in hits {
            if let Some(memory) = self.graph.get_memory(&hit.memory_id).await? {
                results.push(SearchResult {
                    memory: memory.memory,
                    score: hit.score,
                });
            }
        }
        Ok(results)
    }

    /// Relevant prior memories for a user message: semantic top-k with the
    /// graph neighborhood of each hit expanded (one hop).
    pub async fn context(&self, message: &str, top_k: Option<usize>) -> anyhow::Result<Vec<ContextResult>> {
        let hits = self.search(message, top_k.or(Some(3))).await?;
        let mut with_neighbors = Vec::new();
        for hit in hits {
            let related = self
                .graph
                .get_memory(&hit.memory.id)
                .await?
                .map(|m| m.related)
                .unwrap_or_default();
            with_neighbors.push(ContextResult {
                memory: hit.memory,
                score: hit.score,
                related,
            });
        }
        Ok(with_neighbors)
    }

    /// Rebuild the vector index from the graph. Returns number reindexed.
    pub async fn reindex(&self) -> anyhow::Result<usize> {
        let all = self.graph.list_all().await?;
        self.vectors.delete_all().await?;
        for memory in &all {
            let result = self.embeddings.embed(&memory.text).await?;
            self.vectors
                .upsert(&VectorRecord {
                    memory_id: memory.id.clone(),
                    vector: result.vector,
                    model: result.model,
                })
                .await?;
        }
        Ok(all.len())
    }

    /// Warn if vectors in the index were produced by a different model.
    pub async fn embedding_model_drift(&self) -> anyhow::Result<Vec<String>> {
        let current = self.embeddings.model();
        let mut drift = Vec::new();
        for m in self.vectors.models().await? {
            if m != current && !drift.contains(&m) {
                drift.push(m);
            }
        }
        Ok(drift)
    }

    pub async fn close(&self) -> anyhow::Result<()> {
        self.graph.close().await?;
        self.vectors.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::HashEmbeddings;
    use std::sync::Arc;
    use crate::graph::InMemoryGraphStore;
    use crate::vectors::{InMemoryVectorStore, VectorRecord, VectorStore};

    fn make_service() -> (MemoryService, Arc<InMemoryVectorStore>) {
        let vectors = Arc::new(InMemoryVectorStore::new());
        let service = MemoryService::new(
            Arc::new(InMemoryGraphStore::new()),
            vectors.clone(),
            Arc::new(HashEmbeddings::new(64)),
        );
        (service, vectors)
    }

    #[tokio::test]
    async fn store_reindex_search_round_trip_heals_index_drift() {
        let (service, vectors) = make_service();

        let a = service
            .store("alpha stores secrets in vault-7", None)
            .await
            .unwrap();
        service
            .store("the beta cluster is for staging", None)
            .await
            .unwrap();

        // simulate index loss
        vectors.delete_all().await.unwrap();
        assert_eq!(vectors.count().await.unwrap(), 0);

        // reindex rebuilds from the graph
        let n = service.reindex().await.unwrap();
        assert_eq!(n, 2);
        assert_eq!(vectors.count().await.unwrap(), 2);

        let hits = service
            .search("alpha stores secrets in vault-7", None)
            .await
            .unwrap();
        assert_eq!(hits[0].memory.id, a.id);

        // drift detection: index model matches current provider → empty drift
        assert_eq!(service.embedding_model_drift().await.unwrap(), Vec::<String>::new());

        // drift detection: foreign model in index → reported
        vectors
            .upsert(&VectorRecord {
                memory_id: "mem_x".to_string(),
                vector: vec![1.0, 0.0],
                model: "old-model-v1".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(
            service.embedding_model_drift().await.unwrap(),
            vec!["old-model-v1".to_string()]
        );
    }

    #[tokio::test]
    async fn search_hydrates_from_graph_and_skips_orphaned_vectors() {
        let (service, vectors) = make_service();
        let a = service
            .store("deploy server 192.168.0.50", None)
            .await
            .unwrap();
        let embedded = HashEmbeddings::new(64)
            .embed("deploy server 192.168.0.50")
            .await
            .unwrap();
        // orphaned vector: points at a memory that no longer exists in the graph
        vectors
            .upsert(&VectorRecord {
                memory_id: "mem_orphan".to_string(),
                vector: embedded.vector,
                model: "h".to_string(),
            })
            .await
            .unwrap();

        let hits = service
            .search("deploy server 192.168.0.50", Some(5))
            .await
            .unwrap();
        // orphan skipped (not in graph), real memory returned
        assert!(hits.iter().all(|h| h.memory.id != "mem_orphan"));
        assert!(hits.iter().any(|h| h.memory.id == a.id));
    }
}
