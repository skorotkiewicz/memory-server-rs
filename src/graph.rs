//! Graph storage: memories as `(:Memory)` nodes with typed relations in Neo4j.
//! Neo4j is the source of truth; Qdrant is only a rebuildable index.
//!
//! Mirrors packages/memory-server/src/graph.ts.

use async_trait::async_trait;
use neo4rs::{BoltType, Graph, Node, Query};
use serde::Serialize;
use tokio::sync::Mutex as AsyncMutex;
use uuid::Uuid;

pub const ID_PREFIX: &str = "mem_";

/// Serialized with camelCase so the wire format matches the TS server.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MemoryRecord {
    pub id: String,
    pub text: String,
    pub created_at: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RelatedMemory {
    #[serde(flatten)]
    pub memory: MemoryRecord,
    pub relation: String,
    /// "out" or "in"
    pub direction: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MemoryWithRelated {
    #[serde(flatten)]
    pub memory: MemoryRecord,
    pub related: Vec<RelatedMemory>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LinkResult {
    #[serde(rename = "fromId")]
    pub from_id: String,
    #[serde(rename = "toId")]
    pub to_id: String,
    pub relation: String,
}

pub fn new_id() -> String {
    format!("{}{}", ID_PREFIX, Uuid::new_v4())
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Only word chars allowed — the relation becomes a dynamic relationship type.
fn valid_relation(relation: &str) -> bool {
    !relation.is_empty()
        && relation
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[async_trait]
pub trait GraphStore: Send + Sync {
    async fn init(&self) -> anyhow::Result<()>;
    async fn store_memory(&self, text: &str, tags: Option<Vec<String>>) -> anyhow::Result<MemoryRecord>;
    async fn link_memories(&self, from_id: &str, to_id: &str, relation: &str) -> anyhow::Result<LinkResult>;
    async fn get_memory(&self, id: &str) -> anyhow::Result<Option<MemoryWithRelated>>;
    async fn delete_memory(&self, id: &str) -> anyhow::Result<bool>;
    async fn list_all(&self) -> anyhow::Result<Vec<MemoryRecord>>;
    async fn close(&self) -> anyhow::Result<()>;
}

// ---------------------------------------------------------------------------
// Neo4j-backed store
// ---------------------------------------------------------------------------

pub struct Neo4jGraphStore {
    graph: Graph,
}

impl Neo4jGraphStore {
    pub async fn connect(uri: &str, user: &str, password: &str) -> anyhow::Result<Self> {
        // connects eagerly — fails fast if unreachable (like getServerInfo in TS)
        let graph = Graph::new(uri, user, password).await?;
        Ok(Self { graph })
    }

    fn memory_from_node(node: &Node) -> MemoryRecord {
        MemoryRecord {
            id: node.get::<String>("id").unwrap_or_default(),
            text: node.get::<String>("text").unwrap_or_default(),
            created_at: node.get::<String>("createdAt").unwrap_or_default(),
            tags: node
                .get::<Option<Vec<String>>>("tags")
                .ok()
                .flatten()
                .unwrap_or_default(),
        }
    }

    /// OPTIONAL MATCH produces rows with nulls when there are no relations.
    fn related_from_list(list: Vec<BoltType>, direction: &str) -> Vec<RelatedMemory> {
        let mut out = Vec::new();
        for item in list {
            let BoltType::Map(map) = item else { continue };
            let id: Option<String> = map.get::<Option<String>>("id").ok().flatten();
            let Some(id) = id else { continue };
            out.push(RelatedMemory {
                memory: MemoryRecord {
                    id,
                    text: map.get::<Option<String>>("text").ok().flatten().unwrap_or_default(),
                    created_at: map
                        .get::<Option<String>>("createdAt")
                        .ok()
                        .flatten()
                        .unwrap_or_default(),
                    tags: map
                        .get::<Option<Vec<String>>>("tags")
                        .ok()
                        .flatten()
                        .unwrap_or_default(),
                },
                relation: map
                    .get::<Option<String>>("relation")
                    .ok()
                    .flatten()
                    .unwrap_or_default(),
                direction: direction.to_string(),
            });
        }
        out
    }
}

#[async_trait]
impl GraphStore for Neo4jGraphStore {
    async fn init(&self) -> anyhow::Result<()> {
        self.graph
            .run(Query::new(
                "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE"
                    .to_string(),
            ))
            .await?;
        self.graph
            .run(Query::new(
                "CREATE INDEX memory_text IF NOT EXISTS FOR (m:Memory) ON (m.text)".to_string(),
            ))
            .await?;
        Ok(())
    }

    async fn store_memory(&self, text: &str, tags: Option<Vec<String>>) -> anyhow::Result<MemoryRecord> {
        let record = MemoryRecord {
            id: new_id(),
            text: text.to_string(),
            created_at: now_iso(),
            tags: tags.unwrap_or_default(),
        };
        let query = Query::new(
            "CREATE (m:Memory {id: $id, text: $text, createdAt: $createdAt, tags: $tags}) RETURN m"
                .to_string(),
        )
        .param("id", record.id.clone())
        .param("text", record.text.clone())
        .param("createdAt", record.created_at.clone())
        .param("tags", record.tags.clone());
        self.graph.run(query).await?;
        Ok(record)
    }

    async fn link_memories(&self, from_id: &str, to_id: &str, relation: &str) -> anyhow::Result<LinkResult> {
        if !valid_relation(relation) {
            anyhow::bail!("invalid relation name: {}", relation)
        }
        // dynamic relationship type required for typed edges
        let cypher = format!(
            "MATCH (a:Memory {{id: $fromId}}), (b:Memory {{id: $toId}})\n\
             CREATE (a)-[r:{relation}]->(b)\n\
             RETURN a.id AS fromId, b.id AS toId, type(r) AS relation"
        );
        let query = Query::new(cypher)
            .param("fromId", from_id.to_string())
            .param("toId", to_id.to_string());
        let mut stream = self.graph.execute(query).await?;
        match stream.next().await? {
            Some(row) => Ok(LinkResult {
                from_id: row.get::<String>("fromId")?,
                to_id: row.get::<String>("toId")?,
                relation: row.get::<String>("relation")?,
            }),
            None => anyhow::bail!(
                "cannot link: memory \"{}\" or \"{}\" not found",
                from_id,
                to_id
            ),
        }
    }

    async fn get_memory(&self, id: &str) -> anyhow::Result<Option<MemoryWithRelated>> {
        let query = Query::new(
            "MATCH (m:Memory {id: $id})\n\
             OPTIONAL MATCH (m)-[r]->(out:Memory)\n\
             OPTIONAL MATCH (inn:Memory)-[rin]->(m)\n\
             RETURN m, collect(DISTINCT {id: out.id, text: out.text, createdAt: out.createdAt, tags: out.tags, relation: type(r), direction: 'out'}) AS outRel,\n\
                    collect(DISTINCT {id: inn.id, text: inn.text, createdAt: inn.createdAt, tags: inn.tags, relation: type(rin), direction: 'in'}) AS inRel"
                .to_string(),
        )
        .param("id", id.to_string());
        let mut stream = self.graph.execute(query).await?;
        let Some(row) = stream.next().await? else {
            return Ok(None);
        };
        let node = row.get::<Node>("m")?;
        let memory = Self::memory_from_node(&node);
        let out_rel = row.get::<Vec<BoltType>>("outRel").unwrap_or_default();
        let in_rel = row.get::<Vec<BoltType>>("inRel").unwrap_or_default();
        let mut related = Self::related_from_list(out_rel, "out");
        related.extend(Self::related_from_list(in_rel, "in"));
        Ok(Some(MemoryWithRelated { memory, related }))
    }

    async fn delete_memory(&self, id: &str) -> anyhow::Result<bool> {
        // DETACH DELETE removes the node and all its edges, nothing else
        let query = Query::new(
            "MATCH (m:Memory {id: $id}) DETACH DELETE m RETURN count(m) AS n".to_string(),
        )
        .param("id", id.to_string());
        let mut stream = self.graph.execute(query).await?;
        let n = match stream.next().await? {
            Some(row) => row.get::<i64>("n").unwrap_or(0),
            None => 0,
        };
        Ok(n > 0)
    }

    async fn list_all(&self) -> anyhow::Result<Vec<MemoryRecord>> {
        let mut stream = self
            .graph
            .execute(Query::new(
                "MATCH (m:Memory) RETURN m ORDER BY m.createdAt".to_string(),
            ))
            .await?;
        let mut out = Vec::new();
        while let Some(row) = stream.next().await? {
            let node = row.get::<Node>("m")?;
            out.push(Self::memory_from_node(&node));
        }
        Ok(out)
    }

    async fn close(&self) -> anyhow::Result<()> {
        // the neo4rs Graph owns its connection pool; dropped with self
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// In-memory store for unit tests (mirrors Neo4j semantics used above)
// ---------------------------------------------------------------------------

struct InMemoryInner {
    memories: Vec<MemoryRecord>,
    edges: Vec<(String, String, String)>, // (fromId, toId, relation)
}

pub struct InMemoryGraphStore {
    inner: AsyncMutex<InMemoryInner>,
}

impl Default for InMemoryGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryGraphStore {
    pub fn new() -> Self {
        Self {
            inner: AsyncMutex::new(InMemoryInner {
                memories: Vec::new(),
                edges: Vec::new(),
            }),
        }
    }
}

#[async_trait]
impl GraphStore for InMemoryGraphStore {
    async fn init(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn store_memory(&self, text: &str, tags: Option<Vec<String>>) -> anyhow::Result<MemoryRecord> {
        let record = MemoryRecord {
            id: new_id(),
            text: text.to_string(),
            created_at: now_iso(),
            tags: tags.unwrap_or_default(),
        };
        self.inner.lock().await.memories.push(record.clone());
        Ok(record)
    }

    async fn link_memories(&self, from_id: &str, to_id: &str, relation: &str) -> anyhow::Result<LinkResult> {
        let mut inner = self.inner.lock().await;
        if !inner.memories.iter().any(|m| m.id == from_id)
            || !inner.memories.iter().any(|m| m.id == to_id)
        {
            anyhow::bail!("cannot link: memory \"{}\" or \"{}\" not found", from_id, to_id)
        }
        inner
            .edges
            .push((from_id.to_string(), to_id.to_string(), relation.to_string()));
        Ok(LinkResult {
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
            relation: relation.to_string(),
        })
    }

    async fn get_memory(&self, id: &str) -> anyhow::Result<Option<MemoryWithRelated>> {
        let inner = self.inner.lock().await;
        let Some(memory) = inner.memories.iter().find(|m| m.id == id) else {
            return Ok(None);
        };
        let mut related = Vec::new();
        for (from, to, relation) in &inner.edges {
            if from == id {
                if let Some(t) = inner.memories.iter().find(|m| m.id == to) {
                    related.push(RelatedMemory {
                        memory: t.clone(),
                        relation: relation.clone(),
                        direction: "out".to_string(),
                    });
                }
            }
            if to == id {
                if let Some(t) = inner.memories.iter().find(|m| m.id == from) {
                    related.push(RelatedMemory {
                        memory: t.clone(),
                        relation: relation.clone(),
                        direction: "in".to_string(),
                    });
                }
            }
        }
        Ok(Some(MemoryWithRelated {
            memory: memory.clone(),
            related,
        }))
    }

    async fn delete_memory(&self, id: &str) -> anyhow::Result<bool> {
        let mut inner = self.inner.lock().await;
        let before = inner.memories.len();
        inner.memories.retain(|m| m.id != id);
        let deleted = inner.memories.len() < before;
        inner
            .edges
            .retain(|(from, to, _)| from != id && to != id);
        Ok(deleted)
    }

    async fn list_all(&self) -> anyhow::Result<Vec<MemoryRecord>> {
        Ok(self.inner.lock().await.memories.clone())
    }

    async fn close(&self) -> anyhow::Result<()> {
        Ok(())
    }
}
