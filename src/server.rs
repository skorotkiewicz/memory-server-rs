//! MCP server exposing the memory tools over Streamable HTTP.
//! Stateless pattern: each POST is handled independently — simple, safe for
//! concurrent clients, and restarts lose nothing because all state lives in
//! Neo4j/Qdrant.
//!
//! MCP tools: memory_store/search/get/link/delete/context, served stateless
//! over Streamable HTTP (the JSON-RPC surface is implemented directly on axum).

use crate::service::MemoryService;
use anyhow::Result;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::Arc;

pub const SERVER_NAME: &str = "cognitive-memory";
pub const SERVER_VERSION: &str = "0.1.0";
const SERVER_INSTRUCTIONS: &str =
    "Persistent cognitive memory: store, search, link and recall knowledge across sessions.";

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
struct AppState {
    service: Arc<MemoryService>,
    token: Option<Arc<String>>,
}

pub fn tools_list() -> Vec<Value> {
    vec![
        json!({
            "name": "memory_store",
            "description": "Store a new memory (fact, preference, observation) for later semantic recall",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The memory content to store" },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Optional tags" }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "memory_search",
            "description": "Search memories by semantic similarity (paraphrasing works)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to search for" },
                    "top_k": { "type": "integer", "minimum": 1, "maximum": 20, "description": "Max results (default 5)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "memory_get",
            "description": "Get a memory by id, including its linked memories (graph neighborhood)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Memory id" }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "memory_link",
            "description": "Create a typed relation between two memories",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from_id": { "type": "string", "description": "Source memory id" },
                    "to_id": { "type": "string", "description": "Target memory id" },
                    "relation": { "type": "string", "description": "Relation type, e.g. RELATED_TO, CAUSES, PART_OF" }
                },
                "required": ["from_id", "to_id", "relation"]
            }
        }),
        json!({
            "name": "memory_delete",
            "description": "Delete a memory by id (its relations are removed too)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Memory id" }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "memory_context",
            "description": "Get prior memories relevant to the current user message, with graph neighborhoods expanded — call this when a message might relate to previously stored knowledge",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "The current user message" },
                    "top_k": { "type": "integer", "minimum": 1, "maximum": 10, "description": "Max memories (default 3)" }
                },
                "required": ["message"]
            }
        }),
    ]
}

fn json_rpc_result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn json_rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn text_content(value: Value) -> Value {
    json!({ "content": [{ "type": "text", "text": value.to_string() }] })
}

fn text_content_raw(text: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

/// Handle a single JSON-RPC message. Returns None for notifications.
async fn handle_rpc_message(state: &AppState, message: &Value) -> Option<Value> {
    let method = message.get("method").and_then(|m| m.as_str())?.to_string();
    let id = message.get("id").cloned();
    let id = match id {
        Some(v @ (Value::Number(_) | Value::String(_))) => v,
        // notification (or null id) — nothing to respond with
        _ => return None,
    };
    let params = message.get("params").cloned().unwrap_or(json!({}));

    match method.as_str() {
        "initialize" => {
            let pv = params
                .get("protocolVersion")
                .and_then(|v| v.as_str())
                .unwrap_or("2025-06-18");
            Some(json_rpc_result(
                id,
                json!({
                    "protocolVersion": pv,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
                    "instructions": SERVER_INSTRUCTIONS
                }),
            ))
        }
        "ping" => Some(json_rpc_result(id, json!({}))),
        "tools/list" => Some(json_rpc_result(id, json!({ "tools": tools_list() }))),
        "tools/call" => Some(handle_tools_call(state, id, &params).await),
        other => Some(json_rpc_error(id, -32601, &format!("Method not found: {other}"))),
    }
}

async fn handle_tools_call(state: &AppState, id: Value, params: &Value) -> Value {
    let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    macro_rules! arg {
        ($key:expr) => {
            match args.get($key).and_then(|v| v.as_str()) {
                Some(v) => v.to_string(),
                None => {
                    return json_rpc_error(
                        id,
                        -32602,
                        &format!("Invalid arguments: missing required field '{}'", $key),
                    )
                }
            }
        };
    }
    macro_rules! opt_top_k {
        ($key:expr, $max:expr) => {
            match args.get($key) {
                None | Some(Value::Null) => None,
                Some(v) => match v.as_u64() {
                    Some(n) if (1..=$max).contains(&n) => Some(n as usize),
                    _ => {
                        return json_rpc_error(
                            id,
                            -32602,
                            &format!("Invalid arguments: '{}' must be an integer between 1 and {}", $key, $max),
                        )
                    }
                },
            }
        };
    }

    let outcome: Result<Value, String> = match name {
        "memory_store" => {
            let text = arg!("text");
            let tags = args
                .get("tags")
                .and_then(|t| t.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str().map(String::from))
                        .collect::<Vec<String>>()
                });
            state.service.store(&text, tags).await.map_err(|e| format!("{e:#}")).map(|record| text_content(serde_json::to_value(record).unwrap_or_default()))
        }
        "memory_search" => {
            let query = arg!("query");
            let top_k = opt_top_k!("top_k", 20);
            state
                .service
                .search(&query, top_k)
                .await
                .map_err(|e| format!("{e:#}"))
                .and_then(|results| {
                    serde_json::to_value(results).map_err(|e| format!("{e}"))
                })
                .map(text_content)
        }
        "memory_get" => {
            let memory_id = arg!("id");
            match state.service.get(&memory_id).await {
                Ok(Some(memory)) => Ok(text_content(
                    serde_json::to_value(memory).unwrap_or_default(),
                )),
                Ok(None) => Ok(text_content_raw(&format!("memory \"{}\" not found", memory_id))),
                Err(e) => Err(format!("{e:#}")),
            }
        }
        "memory_link" => {
            let from_id = arg!("from_id");
            let to_id = arg!("to_id");
            let relation = arg!("relation");
            state.service.link(&from_id, &to_id, &relation).await.map_err(|e| format!("{e:#}")).map(|link| {
                text_content(serde_json::to_value(link).unwrap_or_default())
            })
        }
        "memory_delete" => {
            let memory_id = arg!("id");
            match state.service.delete(&memory_id).await {
                Ok(true) => Ok(text_content_raw(&format!("deleted {}", memory_id))),
                Ok(false) => Ok(text_content_raw(&format!("memory \"{}\" not found", memory_id))),
                Err(e) => Err(format!("{e:#}")),
            }
        }
        "memory_context" => {
            let message = arg!("message");
            let top_k = opt_top_k!("top_k", 10);
            state
                .service
                .context(&message, top_k)
                .await
                .map_err(|e| format!("{e:#}"))
                .and_then(|results| {
                    serde_json::to_value(results).map_err(|e| format!("{e}"))
                })
                .map(text_content)
        }
        other => {
            return json_rpc_error(id, -32602, &format!("Unknown tool: {other}"));
        }
    };

    match outcome {
        Ok(result) => json_rpc_result(id, result),
        Err(err) => json_rpc_result(
            id,
            json!({ "content": [{ "type": "text", "text": format!("Tool execution failed: {err}") }], "isError": true }),
        ),
    }
}

fn json_response(status: StatusCode, value: Value) -> Response {
    (status, Json(value)).into_response()
}

fn error_json(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({ "error": message })),
    )
        .into_response()
}

async fn health() -> Response {
    json_response(StatusCode::OK, json!({ "ok": true }))
}

/// Handles everything that is not GET /health. Stateless MCP transport:
/// POST (any path) carries JSON-RPC; everything else is rejected.
async fn fallback(State(state): State<AppState>, method: Method, uri: Uri, headers: HeaderMap, body: Body) -> Response {
    match method {
        Method::POST => handle_mcp_post(state, headers, body).await,
        Method::GET | Method::DELETE => error_json(
            StatusCode::METHOD_NOT_ALLOWED,
            "method not allowed in stateless mode; use POST",
        ),
        _ => {
            let _ = uri;
            error_json(StatusCode::METHOD_NOT_ALLOWED, "method not allowed")
        }
    }
}

async fn handle_mcp_post(state: AppState, headers: HeaderMap, body: Body) -> Response {
    let bytes = match axum::body::to_bytes(body, 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return error_json(StatusCode::BAD_REQUEST, "failed to read request body");
        }
    };
    let mut messages: Vec<Value> = match serde_json::from_slice(&bytes) {
        Ok(Value::Array(items)) => items,
        Ok(message @ Value::Object(_)) => vec![message],
        Ok(_) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                json_rpc_error(Value::Null, -32600, "Invalid Request"),
            );
        }
        Err(_) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                json_rpc_error(Value::Null, -32700, "Parse error"),
            );
        }
    };

    // Reject SSE-only requests (stateless JSON mode, like the TS transport with
    // enableJsonResponse: true)
    let accept_ok = headers
        .get(axum::http::header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("application/json"))
        .unwrap_or(true);
    if !accept_ok {
        return error_json(
            StatusCode::NOT_ACCEPTABLE,
            "client must accept application/json",
        );
    }

    if messages.len() == 1 {
        let message = messages.remove(0);
        if message.get("method").is_none() {
            return json_response(
                StatusCode::BAD_REQUEST,
                json_rpc_error(Value::Null, -32600, "Invalid Request"),
            );
        }
        match handle_rpc_message(&state, &message).await {
            Some(response) => json_response(StatusCode::OK, response),
            None => StatusCode::ACCEPTED.into_response(), // notification
        }
    } else {
        let mut responses = Vec::new();
        for message in messages {
            if let Some(response) = handle_rpc_message(&state, &message).await {
                responses.push(response);
            }
        }
        if responses.is_empty() {
            return StatusCode::ACCEPTED.into_response();
        }
        json_response(StatusCode::OK, Value::Array(responses))
    }
}

/// Bearer auth gate, applied to every request before any tool runs.
async fn auth_middleware(
    State(state): State<AppState>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if let Some(token) = &state.token {
        let expected = format!("Bearer {token}");
        let provided = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if provided != expected {
            return error_json(
                StatusCode::UNAUTHORIZED,
                "unauthorized: valid bearer token required",
            );
        }
    }
    next.run(request).await
}

/// Streamable HTTP MCP server. Stateless pattern: each POST gets a fresh
/// handling pass — safe for concurrent clients.
pub async fn start_memory_server(
    service: Arc<MemoryService>,
    options: MemoryServerOptions,
) -> Result<MemoryServerHandle> {
    let state = AppState {
        service,
        token: options.token.map(Arc::new),
    };

    let app = Router::new()
        .route("/health", get(health))
        .fallback(fallback)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", options.port)).await?;
    let port = listener.local_addr()?.port();

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let mut rx = shutdown_rx;
                loop {
                    if *rx.borrow() {
                        break;
                    }
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
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

    async fn rpc(port: u16, body: Value, token: Option<&str>) -> (StatusCode, Value) {
        let url = format!("http://127.0.0.1:{port}/mcp");
        let client = reqwest::Client::new();
        let mut request = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");
        if let Some(tok) = token {
            request = request.bearer_auth(tok);
        }
        let response = request.json(&body).send().await.unwrap();
        let status = response.status();
        let json: Value = response.json().await.unwrap();
        (StatusCode::from_u16(status.as_u16()).unwrap(), json)
    }

    #[tokio::test]
    async fn tool_listing_store_search_round_trip_and_shared_layer() {
        let open = start_test_server(None).await;
        assert!(open.port > 0);

        // --- client 1: initialize + list tools
        let (status, response) = rpc(
            open.port,
            json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "2025-06-18" } }),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(response["result"]["serverInfo"]["name"], "cognitive-memory");

        let (status, response) = rpc(
            open.port,
            json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
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
        assert_eq!(status, StatusCode::OK);
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
        assert_eq!(status, StatusCode::OK);
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
        assert_eq!(status, StatusCode::OK);
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
        assert_eq!(response["result"]["content"][0]["text"], format!("deleted {id}"));

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
        let a: Value = serde_json::from_str(r1["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        let b: Value = serde_json::from_str(r2["result"]["content"][0]["text"].as_str().unwrap()).unwrap();

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
    async fn unknown_tool_and_method_error_cleanly() {
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

        let (_, response) = rpc(
            server.port,
            json!({ "jsonrpc": "2.0", "id": 2, "method": "resources/list" }),
            None,
        )
        .await;
        assert_eq!(response["error"]["code"], -32601);

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
        let (status, response) = rpc(secured.port, json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }), Some("s3cret")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(!response["result"]["tools"].as_array().unwrap().is_empty());

        secured.close().await;
    }
}
