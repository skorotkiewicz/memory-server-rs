# memory-server-rs

The Memory MCP server (Rust).

## Layout

| file | notes |
|---|---|
| `src/graph.rs` | Neo4j `(:Memory)` nodes + typed relations; in-memory store for tests |
| `src/vectors.rs` | Qdrant index (`memories` collection); in-memory store for tests |
| `src/embeddings.rs` | local fastembed (bge-small-en-v1.5), OpenAI-compatible endpoint, hash provider for tests |
| `src/service.rs` | dual-write (graph first, index second), hydration through the graph, reindex, drift detection |
| `src/server.rs` | Streamable HTTP MCP transport (stateless JSON mode) on axum |
| `src/main.rs` | env wiring, `--reindex`, graceful shutdown |

## Env

```
NEO4J_URI      bolt://127.0.0.1:7687
NEO4J_USER / NEO4J_PASSWORD   (or NEO4J_AUTH=neo4j/<pw> from the neo4j image)
QDRANT_URL     http://127.0.0.1:6333     QDRANT_API_KEY
HOST           127.0.0.1 (Docker Compose published address)
PORT           8080
MEMORY_TOKEN   optional bearer auth
ALLOWED_HOSTS  comma-separated extra HTTP Host names for /mcp (localhost allowed by default)
FASTEMBED_CACHE_PATH           persist the local model between restarts
EMBEDDINGS_BASEURL + EMBEDDINGS_MODEL (+ EMBEDDINGS_APIKEY)
               use an external OpenAI-compatible /embeddings endpoint;
               unset = local bge-small-en-v1.5 (384 dims), no content leaves the stack
EMBEDDING_DIMENSIONS           override qdrant collection dimensionality
```

See `.env.example` for the docker-compose variables.

## CLI

```
cargo run -- --reindex    # rebuild the qdrant index from neo4j and exit
```

## Tests

```
cargo test    # unit tests (in-memory stores, no services needed)
```

Integration tests against live Neo4j/Qdrant run when the corresponding env
vars are set (containers are expected to be running, e.g. via
`docker compose up -d neo4j qdrant`):

```
NEO4J_TEST_URI=bolt://127.0.0.1:7687 \
NEO4J_TEST_USER=neo4j NEO4J_TEST_PASSWORD=testpassword \
QDRANT_TEST_URL=http://127.0.0.1:6334 \
cargo test
```

(`QDRANT_TEST_URL` points at the gRPC port, 6334.)

## MCP surface

Stateless Streamable HTTP: `POST /mcp` (JSON-RPC), `GET /health`. Bearer token
required on every request when `MEMORY_TOKEN` is set. Set `ALLOWED_HOSTS` to
allow HTTP access through other hostnames or IPs (for example,
`ALLOWED_HOSTS=memory.example.com,192.168.1.50`). This checks the HTTP Host
header, not CORS or client authorization. Docker Compose publishes on `HOST`
(default `127.0.0.1`); set `HOST=0.0.0.0` for remote clients and use
`MEMORY_TOKEN`. Tools: `memory_store`,
`memory_search`, `memory_get`, `memory_link`, `memory_delete`, `memory_context`
(camelCase records: `id`, `text`, `createdAt`, `tags`, `related`, `score`).
