//! Memory MCP server entrypoint.
//!
//! Env:
//!   NEO4J_URI (default bolt://127.0.0.1:7687)
//!   NEO4J_USER / NEO4J_PASSWORD (default neo4j / from NEO4J_AUTH=neo4j/<pw>)
//!   QDRANT_URL (default http://127.0.0.1:6333), QDRANT_API_KEY
//!   PORT (default 8080), MEMORY_TOKEN (optional bearer auth)
//!   FASTEMBED_CACHE_PATH (persist the local model between restarts)
//!   EMBEDDINGS_BASEURL + EMBEDDINGS_MODEL (+ EMBEDDINGS_APIKEY) for an
//!     external OpenAI-compatible /embeddings endpoint; unset = local model.
//!   EMBEDDING_DIMENSIONS (override qdrant collection dimensionality)
//!
//! CLI: --reindex rebuilds the Qdrant index from Neo4j and exits.

mod embeddings;
mod graph;
mod server;
mod service;
mod vectors;

use embeddings::{create_embedding_provider, EmbeddingsConfig, EmbeddingsProviderKind};
use graph::{GraphStore, Neo4jGraphStore};
use service::MemoryService;
use server::{MemoryServerOptions, MemoryServerHandle};
use std::sync::Arc;
use vectors::QdrantVectorStore;

fn env_or(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.to_string())
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn neo4j_credentials() -> (String, String, String) {
    let uri = env_or("NEO4J_URI", "bolt://127.0.0.1:7687");
    // conventional "user/password" from the neo4j image
    if let Some(auth) = env_nonempty("NEO4J_AUTH")
        && let Some(idx) = auth.find('/')
    {
        return (uri, auth[..idx].to_string(), auth[idx + 1..].to_string());
    }
    (
        uri,
        env_or("NEO4J_USER", "neo4j"),
        env_or("NEO4J_PASSWORD", "password"),
    )
}

async fn build_service() -> anyhow::Result<Arc<MemoryService>> {
    let (uri, user, password) = neo4j_credentials();
    let graph = Neo4jGraphStore::connect(&uri, &user, &password).await?;
    graph.init().await?;

    let dimensions = std::env::var("EMBEDDING_DIMENSIONS")
        .ok()
        .and_then(|d| d.parse().ok());
    let vectors = QdrantVectorStore::new(
        &env_or("QDRANT_URL", "http://127.0.0.1:6333"),
        env_nonempty("QDRANT_API_KEY"),
        dimensions,
    )?;

    // embeddings: env-only. Unset = local bge-small-en-v1.5 (baked into the image);
    // set EMBEDDINGS_BASEURL + EMBEDDINGS_MODEL for an external OpenAI-compatible
    // /embeddings endpoint.
    let config = EmbeddingsConfig {
        provider: if env_nonempty("EMBEDDINGS_BASEURL").is_some()
            && env_nonempty("EMBEDDINGS_MODEL").is_some()
        {
            EmbeddingsProviderKind::OpenAICompatible
        } else {
            EmbeddingsProviderKind::Local
        },
        baseurl: env_nonempty("EMBEDDINGS_BASEURL"),
        apikey: env_nonempty("EMBEDDINGS_APIKEY"),
        model: env_nonempty("EMBEDDINGS_MODEL"),
    };
    let embeddings = create_embedding_provider(&config);

    Ok(Arc::new(MemoryService::new(
        Arc::new(graph),
        Arc::new(vectors),
        Arc::from(embeddings),
    )))
}

async fn sigterm() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    }
    #[cfg(not(unix))]
    std::future::pending::<()>().await;
}

async fn run() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--reindex") {
        println!("reindexing qdrant from neo4j...");
        let service = build_service().await?;
        let n = service.reindex().await?;
        service.close().await?;
        println!("reindexed {n} memories");
        return Ok(());
    }

    let service = build_service().await?;

    let drift = service.embedding_model_drift().await.unwrap_or_default();
    if !drift.is_empty() {
        eprintln!(
            "WARNING: qdrant index contains vectors from other embedding models ({}); \
             current model is different. Run with --reindex to rebuild the index.",
            drift.join(", ")
        );
    }

    let port: u16 = env_or("PORT", "8080").parse()?;
    let token = env_nonempty("MEMORY_TOKEN");
    let handle: MemoryServerHandle = server::start_memory_server(
        service.clone(),
        MemoryServerOptions {
            port,
            token: token.clone(),
        },
    )
    .await?;
    println!(
        "memory MCP server listening on :{} ({})",
        handle.port,
        if token.is_some() {
            "bearer auth ON"
        } else {
            "no auth"
        }
    );

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = sigterm() => {},
    }

    handle.close().await;
    service.close().await?;
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("memory-server failed to start: {err:#}");
        std::process::exit(1);
    }
}
