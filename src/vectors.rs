//! Vector storage: memory embeddings in Qdrant, keyed by Neo4j memory id.
//! The index is rebuildable at any time from the graph (reindex()).
//!
use async_trait::async_trait;
use md5::{Digest, Md5};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointId,
    PointStruct, PointsIdsList, QueryPointsBuilder, ScrollPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use std::collections::BTreeSet;
use std::sync::Mutex;
use uuid::Uuid;

pub const COLLECTION_NAME: &str = "memories";

#[derive(Debug, Clone)]
pub struct VectorRecord {
    /// Neo4j memory id — the payload carries it so hydration goes through the graph
    pub memory_id: String,
    pub vector: Vec<f32>,
    /// embedding model name, recorded per point to detect mismatches
    pub model: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VectorSearchHit {
    #[serde(rename = "memoryId")]
    pub memory_id: String,
    pub score: f32,
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    /// creates the collection if missing; no-op otherwise
    #[allow(dead_code)]
    async fn init(&self) -> anyhow::Result<()>;
    async fn upsert(&self, record: &VectorRecord) -> anyhow::Result<()>;
    async fn search(&self, vector: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorSearchHit>>;
    async fn delete(&self, memory_id: &str) -> anyhow::Result<()>;
    async fn delete_all(&self) -> anyhow::Result<()>;
    #[allow(dead_code)]
    async fn count(&self) -> anyhow::Result<u64>;
    /// embedding model names present in the index (for drift detection)
    async fn models(&self) -> anyhow::Result<Vec<String>>;
    async fn close(&self) -> anyhow::Result<()>;
}

/// Qdrant point ids must be unsigned ints or UUIDs. Our Neo4j ids are
/// `mem_<uuid>`; strip the prefix to reuse the uuid part deterministically.
pub fn uuid_to_ulid(memory_id: &str) -> String {
    let uuid = memory_id.strip_prefix("mem_").unwrap_or(memory_id);
    if let Ok(u) = Uuid::parse_str(uuid) {
        return u.to_string();
    }
    // deterministic fallback: hash into a uuid-shaped id
    let digest = Md5::digest(memory_id.as_bytes());
    let h: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

fn point_id(memory_id: &str) -> PointId {
    // uuid_to_ulid always returns a uuid-shaped string
    PointId::from(uuid_to_ulid(memory_id))
}

pub struct QdrantVectorStore {
    client: Qdrant,
    /// only read by init() — integration tests set it explicitly
    #[allow(dead_code)]
    dimensions: Option<u32>,
    collection_ready: Mutex<bool>,
}

impl QdrantVectorStore {
    pub fn new(url: &str, api_key: Option<String>, dimensions: Option<u32>) -> anyhow::Result<Self> {
        let mut builder = Qdrant::from_url(url);
        if let Some(key) = api_key {
            builder = builder.api_key(key);
        }
        let client = builder.build()?;
        Ok(Self {
            client,
            dimensions,
            collection_ready: Mutex::new(false),
        })
    }

    /// create the collection on first use, auto-detecting dimensions from the vector
    async fn ensure_collection(&self, dims: usize) -> anyhow::Result<()> {
        if *self.collection_ready.lock().unwrap() {
            return Ok(());
        }
        if !self.collection_exists().await? {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(COLLECTION_NAME)
                        .vectors_config(VectorParamsBuilder::new(dims as u64, Distance::Cosine)),
                )
                .await?;
        }
        *self.collection_ready.lock().unwrap() = true;
        Ok(())
    }

    async fn collection_exists(&self) -> anyhow::Result<bool> {
        Ok(self.client.collection_exists(COLLECTION_NAME).await?)
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn init(&self) -> anyhow::Result<()> {
        if !self.collection_exists().await? {
            let dims = self.dimensions.ok_or_else(|| {
                anyhow::anyhow!(
                    "cannot create qdrant collection: embedding dimensions unknown \
                     (embed one memory first or set embeddings.dimensions in config)"
                )
            })?;
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(COLLECTION_NAME)
                        .vectors_config(VectorParamsBuilder::new(dims as u64, Distance::Cosine)),
                )
                .await?;
        }
        *self.collection_ready.lock().unwrap() = true;
        Ok(())
    }

    async fn upsert(&self, record: &VectorRecord) -> anyhow::Result<()> {
        self.ensure_collection(record.vector.len()).await?;
        let payload: Payload = serde_json::json!({
            "memoryId": record.memory_id,
            "model": record.model,
        })
        .try_into()?;
        let point = PointStruct::new(point_id(&record.memory_id), record.vector.clone(), payload);
        self.client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, vec![point]))
            .await?;
        Ok(())
    }

    async fn search(&self, vector: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorSearchHit>> {
        // fresh deployment: no collection yet → nothing to find (spec: empty result, no error)
        if !self.collection_exists().await? {
            return Ok(vec![]);
        }
        let result = self
            .client
            .query(
                QueryPointsBuilder::new(COLLECTION_NAME)
                    .query(vector.to_vec())
                    .limit(top_k as u64)
                    .with_payload(true),
            )
            .await?;
        Ok(result
            .result
            .iter()
            .filter_map(|point| {
                let memory_id = payload_string(&point.payload, "memoryId")?;
                if memory_id.is_empty() {
                    return None;
                }
                Some(VectorSearchHit {
                    memory_id,
                    score: point.score,
                })
            })
            .collect())
    }

    async fn delete(&self, memory_id: &str) -> anyhow::Result<()> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(COLLECTION_NAME)
                    .points(PointsIdsList::from(vec![point_id(memory_id)])),
            )
            .await?;
        Ok(())
    }

    async fn delete_all(&self) -> anyhow::Result<()> {
        // empty filter matches every point
        self.client
            .delete_points(
                DeletePointsBuilder::new(COLLECTION_NAME).points(Filter::default()),
            )
            .await?;
        Ok(())
    }

    async fn count(&self) -> anyhow::Result<u64> {
        let info = self.client.collection_info(COLLECTION_NAME).await?;
        Ok(info.result.and_then(|r| r.points_count).unwrap_or(0))
    }

    async fn models(&self) -> anyhow::Result<Vec<String>> {
        let response = self
            .client
            .scroll(
                ScrollPointsBuilder::new(COLLECTION_NAME)
                    .with_payload(true)
                    .limit(1000),
            )
            .await?;
        let mut names = BTreeSet::new();
        for point in &response.result {
            if let Some(model) = payload_string(&point.payload, "model") {
                names.insert(model);
            }
        }
        Ok(names.into_iter().collect())
    }

    async fn close(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

fn payload_string(
    payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
) -> Option<String> {
    let value = payload.get(key)?;
    match value.kind.as_ref()? {
        qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    }
}

/// Deterministic in-memory vector store for unit tests.
#[allow(dead_code)]
pub struct InMemoryVectorStore {
    points: Mutex<Vec<VectorRecord>>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            points: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn init(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn upsert(&self, record: &VectorRecord) -> anyhow::Result<()> {
        let mut points = self.points.lock().unwrap();
        match points.iter_mut().find(|p| p.memory_id == record.memory_id) {
            Some(existing) => *existing = record.clone(),
            None => points.push(record.clone()),
        }
        Ok(())
    }

    async fn search(&self, vector: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorSearchHit>> {
        let mut hits: Vec<VectorSearchHit> = self
            .points
            .lock()
            .unwrap()
            .iter()
            .map(|rec| VectorSearchHit {
                memory_id: rec.memory_id.clone(),
                score: cosine_similarity(vector, &rec.vector),
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(top_k);
        Ok(hits)
    }

    async fn delete(&self, memory_id: &str) -> anyhow::Result<()> {
        self.points.lock().unwrap().retain(|p| p.memory_id != memory_id);
        Ok(())
    }

    async fn delete_all(&self) -> anyhow::Result<()> {
        self.points.lock().unwrap().clear();
        Ok(())
    }

    async fn count(&self) -> anyhow::Result<u64> {
        Ok(self.points.lock().unwrap().len() as u64)
    }

    async fn models(&self) -> anyhow::Result<Vec<String>> {
        let mut names = BTreeSet::new();
        for p in self.points.lock().unwrap().iter() {
            names.insert(p.model.clone());
        }
        Ok(names.into_iter().collect())
    }

    async fn close(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[allow(dead_code)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    dot / if denom != 0.0 { denom } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_vector(seed: u32, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| ((seed * (i as u32 + 3)) % 7) as f32 / 7.0)
            .collect()
    }

    mod unit {
        use super::*;

        #[test]
        fn strips_mem_prefix_deterministically() {
            let id = "mem_123e4567-e89b-42d3-a456-426614174000";
            assert_eq!(
                uuid_to_ulid(id),
                "123e4567-e89b-42d3-a456-426614174000"
            );
            assert_eq!(uuid_to_ulid(id), uuid_to_ulid(id));
        }

        #[test]
        fn non_uuid_ids_get_a_deterministic_uuid_shaped_hash() {
            let out = uuid_to_ulid("weird-id");
            assert!(Uuid::parse_str(&out).is_ok(), "not uuid-shaped: {out}");
            assert_eq!(out, uuid_to_ulid("weird-id"));
        }
    }

    // Integration tests against a live Qdrant.
    // Skipped unless QDRANT_TEST_URL is set (gRPC port, e.g.
    // docker run -p 127.0.0.1:6334:6334 qdrant/qdrant)
    mod integration {
        use super::*;

        fn url() -> Option<String> {
            std::env::var("QDRANT_TEST_URL").ok().filter(|u| !u.is_empty())
        }

        /// serialize integration tests — they share one scratch collection
        static QDRANT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

        async fn make_store() -> (QdrantVectorStore, tokio::sync::MutexGuard<'static, ()>) {
            let guard = QDRANT_LOCK.lock().await;
            let store = QdrantVectorStore::new(&url().unwrap(), None, Some(4)).unwrap();
            store.init().await.unwrap();
            store.delete_all().await.unwrap();
            (store, guard)
        }

        #[tokio::test]
        async fn init_is_idempotent() {
            let url = match url() {
                Some(url) => url,
                None => return, // skipped: no QDRANT_TEST_URL
            };
            let store = QdrantVectorStore::new(&url, None, Some(4)).unwrap();
            store.init().await.unwrap();
            store.init().await.unwrap(); // second call must not throw
        }

        #[tokio::test]
        async fn upsert_search_delete_round_trip() {
            if url().is_none() {
                return; // skipped: no QDRANT_TEST_URL
            }
            let (store, _guard) = make_store().await;
            for (id, seed) in [
                ("mem_00000000-0000-4000-8000-000000000001", 1),
                ("mem_00000000-0000-4000-8000-000000000002", 1),
                ("mem_00000000-0000-4000-8000-000000000003", 5),
            ] {
                store
                    .upsert(&VectorRecord {
                        memory_id: id.to_string(),
                        vector: fake_vector(seed, 4),
                        model: "m1".to_string(),
                    })
                    .await
                    .unwrap();
            }

            assert_eq!(store.count().await.unwrap(), 3);

            let hits = store.search(&fake_vector(1, 4), 2).await.unwrap();
            assert_eq!(hits.len(), 2);
            // exact-match vectors score highest
            assert!(hits[0].memory_id.ends_with("0001") || hits[0].memory_id.ends_with("0002"));
            assert!(hits[0].score > 0.99);

            store
                .delete("mem_00000000-0000-4000-8000-000000000001")
                .await
                .unwrap();
            assert_eq!(store.count().await.unwrap(), 2);
        }

        #[tokio::test]
        async fn empty_search_returns_empty_result() {
            if url().is_none() {
                return; // skipped: no QDRANT_TEST_URL
            }
            let (store, _guard) = make_store().await;
            let hits = store.search(&fake_vector(1, 4), 5).await.unwrap();
            assert!(hits.is_empty());
        }
    }
}
