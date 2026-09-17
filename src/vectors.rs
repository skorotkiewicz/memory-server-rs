//! Vector storage: memory embeddings in Qdrant, keyed by Neo4j memory id.
//! The index is rebuildable at any time from the graph (reindex()).
//!
//! Mirrors packages/memory-server/src/vectors.ts.

use async_trait::async_trait;
use md5::{Digest, Md5};
use qdrant_client::qdrant::{
    points_selector, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointId,
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
    async fn init(&self) -> anyhow::Result<()>;
    async fn upsert(&self, record: &VectorRecord) -> anyhow::Result<()>;
    async fn search(&self, vector: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorSearchHit>>;
    async fn delete(&self, memory_id: &str) -> anyhow::Result<()>;
    async fn delete_all(&self) -> anyhow::Result<()>;
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
    // uuid_to_ulid always returns a parseable uuid
    let u = Uuid::parse_str(&uuid_to_ulid(memory_id)).expect("uuid-shaped id");
    PointId::from(u)
}

pub struct QdrantVectorStore {
    client: Qdrant,
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
                let memory_id = payload_string(point, "memoryId")?;
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
            if let Some(model) = payload_string(point, "model") {
                names.insert(model);
            }
        }
        Ok(names.into_iter().collect())
    }

    async fn close(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

fn payload_string(point: &qdrant_client::qdrant::ScoredPoint, key: &str) -> Option<String> {
    let value = point.payload.get(key)?;
    match &value.kind? {
        qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    }
}

/// Deterministic in-memory vector store for unit tests.
pub struct InMemoryVectorStore {
    points: Mutex<Vec<VectorRecord>>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

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
