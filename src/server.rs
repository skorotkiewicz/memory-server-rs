//! MCP server exposing the memory tools over Streamable HTTP, built on the
//! official Rust MCP SDK (rmcp).
//!
//! Stateless JSON mode: each POST is handled independently — simple, safe for
//! concurrent clients, and restarts lose nothing because all state lives in
//! Neo4j/Qdrant.
//!
//! Tools: memory_store / memory_search / memory_get / memory_link /
//! memory_delete / memory_context.

use crate::service::MemoryService;
use anyhow::Result;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ErrorData};
use rmcp::{ServerHandler, schemars, tool, tool_handler, tool_router};
use serde_json::json;
use std::sync::Arc;

pub struct MemoryServerOptions {
    pub port: u16,
    /// when set, requests without this bearer token are rejected before any tool runs
    pub token: Option<String>,
}

pub struct MemoryServerHandle {
    pub port: u16,
    shutdown: tokio::sync::watch::Sender<bool>,
}

impl MemoryServerHandle {
    pub async fn close(&self) {
        let _ = self.shutdown.send(true);
        // give the graceful shutdown a moment to drain
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

#[derive(Clone)]
pub struct MemoryMcpServer {
    service: Arc<MemoryService>,
    tool_router: ToolRouter<Self>,
}

// ---- tool arguments (snake_case, same names as the clients send) ----------

macro_rules! top_k_bounds {
    ($v:expr, $max:expr) => {
        match $v {
            Some(n) if (1..=$max).contains(&n) => Some(n as usize),
            Some(n) => {
                return Err(ErrorData::invalid_params(
                    format!("top_k must be between 1 and {}, got {n}", $max),
                    None,
                ))
            }
            None => None,
        }
    };
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StoreParams {
    /// The memory content to store
    pub text: String,
    /// Optional tags
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// What to search for
    pub query: String,
    /// Max results (default 5)
    pub top_k: Option<u64>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetParams {
    /// Memory id
    pub id: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct LinkParams {
    /// Source memory id
    pub from_id: String,
    /// Target memory id
    pub to_id: String,
    /// Relation type, e.g. RELATED_TO, CAUSES, PART_OF
    pub relation: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteParams {
    /// Memory id
    pub id: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ContextParams {
    /// The current user message
    pub message: String,
    /// Max memories (default 3)
    pub top_k: Option<u64>,
}

fn internal_error(err: anyhow::Error) -> ErrorData {
    ErrorData::internal_error(format!("{err:#}"), None)
}

#[tool_router]
impl MemoryMcpServer {
    pub fn new(service: Arc<MemoryService>) -> Self {
        Self {
            service,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Store a new memory (fact, preference, observation) for later semantic recall"
    )]
    async fn memory_store(
        &self,
        Parameters(params): Parameters<StoreParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let record = self
            .service
            .store(&params.text, params.tags)
            .await
            .map_err(internal_error)?;
        let json = serde_json::to_string(&record)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
    }

    #[tool(description = "Search memories by semantic similarity (paraphrasing works)")]
    async fn memory_search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let top_k = top_k_bounds!(params.top_k, 20);
        let results = self
            .service
            .search(&params.query, top_k)
            .await
            .map_err(internal_error)?;
        let json = serde_json::to_string(&results)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
    }

    #[tool(description = "Get a memory by id, including its linked memories (graph neighborhood)")]
    async fn memory_get(
        &self,
        Parameters(params): Parameters<GetParams>,
    ) -> Result<CallToolResult, ErrorData> {
        match self.service.get(&params.id).await.map_err(internal_error)? {
            Some(memory) => {
                let json = serde_json::to_string(&memory)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
            }
            None => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "memory \"{}\" not found",
                params.id
            ))])),
        }
    }

    #[tool(description = "Create a typed relation between two memories")]
    async fn memory_link(
        &self,
        Parameters(params): Parameters<LinkParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let link = self
            .service
            .link(&params.from_id, &params.to_id, &params.relation)
            .await
            .map_err(internal_error)?;
        let json = serde_json::to_string(&link)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
    }

    #[tool(description = "Delete a memory by id (its relations are removed too)")]
    async fn memory_delete(
        &self,
        Parameters(params): Parameters<DeleteParams>,
    ) -> Result<CallToolResult, ErrorData> {
        match self
            .service
            .delete(&params.id)
            .await
            .map_err(internal_error)?
        {
            true => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "deleted {}",
                params.id
            ))])),
            false => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "memory \"{}\" not found",
                params.id
            ))])),
        }
    }

    #[tool(
        description = "Get prior memories relevant to the current user message, with graph neighborhoods expanded — call this when a message might relate to previously stored knowledge"
    )]
    async fn memory_context(
        &self,
        Parameters(params): Parameters<ContextParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let top_k = top_k_bounds!(params.top_k, 10);
        let results = self
            .service
            .context(&params.message, top_k)
            .await
            .map_err(internal_error)?;
        let json = serde_json::to_string(&results)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
    }
}

#[tool_handler(
    router = self.tool_router,
    name = "cognitive-memory",
    version = "0.1.0",
    instructions = "Persistent cognitive memory: store, search, link and recall knowledge across sessions."
)]
impl ServerHandler for MemoryMcpServer {}

// ---- HTTP wiring -----------------------------------------------------------

#[derive(Clone)]
struct AppState {
    token: Option<Arc<String>>,
}

fn json_response(
    status: axum::http::StatusCode,
    value: serde_json::Value,
) -> axum::response::Response {
    (status, Json(value)).into_response()
}

async fn health() -> axum::response::Response {
    json_response(axum::http::StatusCode::OK, json!({ "ok": true }))
}

/// Bearer auth gate, applied to every request before any tool runs.
async fn auth_middleware(
    State(state): State<AppState>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    if let Some(token) = &state.token {
        let expected = format!("Bearer {token}");
        let provided = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if provided != expected {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "unauthorized: valid bearer token required" })),
            )
                .into_response();
        }
    }
    next.run(request).await
}

/// Streamable HTTP MCP server. Stateless JSON mode: each POST is handled
/// independently — safe for concurrent clients.
pub async fn start_memory_server(
    service: Arc<MemoryService>,
    options: MemoryServerOptions,
) -> Result<MemoryServerHandle> {
    use rmcp::transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    };

    let cancellation = tokio_util::sync::CancellationToken::new();
    let mcp_service: StreamableHttpService<MemoryMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            {
                let service = service.clone();
                move || Ok(MemoryMcpServer::new(service.clone()))
            },
            Default::default(), // LocalSessionManager
            StreamableHttpServerConfig::default()
                .with_legacy_session_mode(false)
                .with_json_response(true)
                .with_cancellation_token(cancellation.child_token()),
        );

    let state = AppState {
        token: options.token.map(Arc::new),
    };

    let app = Router::new()
        .route("/health", get(health))
        .nest_service("/mcp", mcp_service)
        .fallback((
            axum::http::StatusCode::METHOD_NOT_ALLOWED,
            Json(json!({ "error": "method not allowed in stateless mode; use POST" })),
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", options.port)).await?;
    let port = listener.local_addr()?.port();

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let shutdown_token = cancellation.clone();
    tokio::spawn(async move {
        let mut rx = shutdown_rx;
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                loop {
                    if *rx.borrow() {
                        break;
                    }
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
                shutdown_token.cancel();
            })
            .await;
    });

    Ok(MemoryServerHandle {
        port,
        shutdown: shutdown_tx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::HashEmbeddings;
    use crate::graph::InMemoryGraphStore;
    use crate::vectors::InMemoryVectorStore;
    use serde_json::Value;

    fn make_service() -> Arc<MemoryService> {
        Arc::new(MemoryService::new(
            Arc::new(InMemoryGraphStore::new()),
            Arc::new(InMemoryVectorStore::new()),
            Arc::new(HashEmbeddings::new(64)),
        ))
    }

    async fn start_test_server(token: Option<&str>) -> MemoryServerHandle {
        start_memory_server(
            make_service(),
            MemoryServerOptions {
                port: 0,
                token: token.map(String::from),
            },
        )
        .await
        .unwrap()
    }

    async fn rpc(port: u16, body: Value, token: Option<&str>) -> (axum::http::StatusCode, Value) {
        let url = format!("http://127.0.0.1:{port}/mcp");
        let client = reqwest::Client::new();
        let mut request = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        if let Some(tok) = token {
            request = request.bearer_auth(tok);
        }
        let response = request.json(&body).send().await.unwrap();
        let status = response.status();
        let text = response.text().await.unwrap();
        let json: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("non-JSON response ({status}): {e} — body: {text:.300}"));

        (
            axum::http::StatusCode::from_u16(status.as_u16()).unwrap(),
            json,
        )
    }

    #[tokio::test]
    async fn tool_listing_store_search_round_trip_and_shared_layer() {
        let open = start_test_server(None).await;
        assert!(open.port > 0);

        // --- client 1: initialize + list tools
        let (status, response) = rpc(
            open.port,
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": { "name": "test-harness", "version": "0.0.1" }
                }
            }),
            None,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(response["result"]["serverInfo"]["name"], "cognitive-memory");

        let (status, response) = rpc(
            open.port,
            json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
            None,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let mut names: Vec<String> = response["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "memory_context",
                "memory_delete",
                "memory_get",
                "memory_link",
                "memory_search",
                "memory_store",
            ]
        );

        // --- client 1 stores a memory
        let (status, response) = rpc(
            open.port,
            json!({
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": { "name": "memory_store", "arguments": { "text": "the deploy server is at 192.168.0.50", "tags": ["infra"] } }
            }),
            None,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let stored: Value =
            serde_json::from_str(response["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert!(stored["id"].as_str().unwrap().starts_with("mem_"));

        // --- client 2 (simulating an external harness) finds it — one shared layer
        let (status, response) = rpc(
            open.port,
            json!({
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": { "name": "memory_search", "arguments": { "query": "the deploy server is at 192.168.0.50" } }
            }),
            None,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let found: Value =
            serde_json::from_str(response["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(found.as_array().unwrap().len(), 1);
        assert_eq!(found[0]["memory"]["id"], stored["id"]);

        // --- context tool returns the memory for a related message
        let (status, response) = rpc(
            open.port,
            json!({
                "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": { "name": "memory_context", "arguments": { "message": "the deploy server is at 192.168.0.50" } }
            }),
            None,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let ctx: Value =
            serde_json::from_str(response["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(ctx.as_array().unwrap().len(), 1);

        open.close().await;
    }

    #[tokio::test]
    async fn get_and_delete_round_trip() {
        let server = start_test_server(None).await;

        let (_, response) = rpc(
            server.port,
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": { "name": "memory_store", "arguments": { "text": "k8s cluster prod-1" } }
            }),
            None,
        )
        .await;
        let stored: Value =
            serde_json::from_str(response["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        let id = stored["id"].as_str().unwrap().to_string();

        // get
        let (_, response) = rpc(
            server.port,
            json!({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": { "name": "memory_get", "arguments": { "id": id } }
            }),
            None,
        )
        .await;
        let memory: Value =
            serde_json::from_str(response["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(memory["text"], "k8s cluster prod-1");
        assert_eq!(memory["tags"].as_array().map(|a| a.len()), Some(0));

        // get missing
        let (_, response) = rpc(
            server.port,
            json!({
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": { "name": "memory_get", "arguments": { "id": "mem_missing" } }
            }),
            None,
        )
        .await;
        assert_eq!(
            response["result"]["content"][0]["text"],
            "memory \"mem_missing\" not found"
        );

        // delete
        let (_, response) = rpc(
            server.port,
            json!({
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": { "name": "memory_delete", "arguments": { "id": id } }
            }),
            None,
        )
        .await;
        assert_eq!(
            response["result"]["content"][0]["text"],
            format!("deleted {id}")
        );

        // delete again → not found
        let (_, response) = rpc(
            server.port,
            json!({
                "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": { "name": "memory_delete", "arguments": { "id": id } }
            }),
            None,
        )
        .await;
        assert_eq!(
            response["result"]["content"][0]["text"],
            format!("memory \"{id}\" not found")
        );

        server.close().await;
    }

    #[tokio::test]
    async fn link_and_context_round_trip() {
        let server = start_test_server(None).await;

        let store = |text: &str, id: i32| {
            rpc(
                server.port,
                json!({
                    "jsonrpc": "2.0", "id": id, "method": "tools/call",
                    "params": { "name": "memory_store", "arguments": { "text": text } }
                }),
                None,
            )
        };
        let (_, r1) = store("rust rewrite done", 1).await;
        let (_, r2) = store("fastembed ported to rust", 2).await;
        let a: Value =
            serde_json::from_str(r1["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        let b: Value =
            serde_json::from_str(r2["result"]["content"][0]["text"].as_str().unwrap()).unwrap();

        let (_, response) = rpc(
            server.port,
            json!({
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": { "name": "memory_link", "arguments": {
                    "from_id": a["id"], "to_id": b["id"], "relation": "RELATED_TO" } }
            }),
            None,
        )
        .await;
        let link: Value =
            serde_json::from_str(response["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(link["relation"], "RELATED_TO");
        assert_eq!(link["fromId"], a["id"]);
        assert_eq!(link["toId"], b["id"]);

        // get expands the neighborhood
        let (_, response) = rpc(
            server.port,
            json!({
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": { "name": "memory_get", "arguments": { "id": a["id"] } }
            }),
            None,
        )
        .await;
        let memory: Value =
            serde_json::from_str(response["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(memory["related"].as_array().unwrap().len(), 1);
        assert_eq!(memory["related"][0]["direction"], "out");

        server.close().await;
    }

    #[tokio::test]
    async fn unknown_tool_errors_cleanly() {
        let server = start_test_server(None).await;

        let (_, response) = rpc(
            server.port,
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": { "name": "no_such_tool", "arguments": {} }
            }),
            None,
        )
        .await;
        assert_eq!(response["error"]["code"], -32602);

        server.close().await;
    }

    #[tokio::test]
    async fn health_endpoint() {
        let server = start_test_server(None).await;
        let client = reqwest::Client::new();
        let response = client
            .get(format!("http://127.0.0.1:{}/health", server.port))
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["ok"], true);
        server.close().await;
    }

    #[tokio::test]
    async fn bearer_token_unauthenticated_rejected_before_any_tool_runs() {
        let secured = start_test_server(Some("s3cret")).await;

        // raw request without/with-wrong token → 401, never reaches tools
        for token in [None, Some("wrong")] {
            let client = reqwest::Client::new();
            let mut request = client
                .post(format!("http://127.0.0.1:{}/mcp", secured.port))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream")
                .json(&json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }));
            if let Some(tok) = token {
                request = request.bearer_auth(tok);
            }
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), 401);
        }

        // health is also behind the token
        let client = reqwest::Client::new();
        let response = client
            .get(format!("http://127.0.0.1:{}/health", secured.port))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401);

        // client with the token works
        let (status, response) = rpc(
            secured.port,
            json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }),
            Some("s3cret"),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(!response["result"]["tools"].as_array().unwrap().is_empty());

        secured.close().await;
    }
}
