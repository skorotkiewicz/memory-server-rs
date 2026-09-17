//! Embedding providers:
//! - local fastembed (bge-small-en-v1.5) — self-hosted, no memory content leaves the stack
//! - OpenAI-compatible /embeddings endpoint — user-configured
//! - hash embeddings — deterministic fake for unit tests
//!
use async_trait::async_trait;
use md5::{Digest, Md5};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::OnceCell;

#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    pub vector: Vec<f32>,
    pub model: String,
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// embed one text; model name is recorded per vector for mismatch detection
    async fn embed(&self, text: &str) -> anyhow::Result<EmbeddingResult>;
    /// model identifier recorded alongside vectors
    fn model(&self) -> String;
}

/// OpenAI-compatible /embeddings endpoint (user-configured).
pub struct OpenAICompatibleEmbeddings {
    baseurl: String,
    apikey: Option<String>,
    model: String,
    http: reqwest::Client,
}

impl OpenAICompatibleEmbeddings {
    pub fn new(baseurl: &str, apikey: &str, model: &str) -> Self {
        Self {
            baseurl: baseurl.to_string(),
            apikey: if apikey.is_empty() {
                None
            } else {
                Some(apikey.to_string())
            },
            model: model.to_string(),
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAICompatibleEmbeddings {
    async fn embed(&self, text: &str) -> anyhow::Result<EmbeddingResult> {
        let url = format!("{}/embeddings", self.baseurl.trim_end_matches('/'));
        let mut request = self
            .http
            .post(&url)
            .json(&serde_json::json!({ "model": self.model, "input": text }));
        if let Some(key) = &self.apikey {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|e| anyhow::anyhow!("{e}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let mut short = body.chars().take(200).collect::<String>();
            if !body.is_empty() && short.len() < body.len() {
                short.push('…');
            }
            anyhow::bail!(
                "embedding endpoint error {} for model \"{}\": {}",
                status.as_u16(),
                self.model,
                short
            )
        }
        let data: serde_json::Value = response.json().await?;
        let vector = data
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|d| d.get("embedding"))
            .and_then(|e| e.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Vec<f32>>()
            });
        match vector {
            Some(v) => Ok(EmbeddingResult {
                vector: v,
                model: self.model.clone(),
            }),
            None => anyhow::bail!("embedding endpoint returned no embedding vector"),
        }
    }

    fn model(&self) -> String {
        self.model.clone()
    }
}

/// Local semantic embedding model (fastembed / bge-small-en-v1.5, 384 dims).
/// Runs fully inside the memory-server process — no network calls at inference
/// time. The model file is downloaded once and cached (respects
/// FASTEMBED_CACHE_PATH so Docker can persist it in a volume).
pub struct LocalEmbeddings {
    embedder: OnceCell<Arc<std::sync::Mutex<fastembed::TextEmbedding>>>,
    cache_dir: Option<PathBuf>,
}

impl LocalEmbeddings {
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        Self {
            embedder: OnceCell::new(),
            cache_dir,
        }
    }

    async fn ensure_init(
        &self,
    ) -> anyhow::Result<Arc<std::sync::Mutex<fastembed::TextEmbedding>>> {
        self.embedder
            .get_or_try_init(|| async {
                let cache_dir = self.cache_dir.clone();
                let init = tokio::task::spawn_blocking(move || {
                    let mut options =
                        fastembed::TextInitOptions::new(fastembed::EmbeddingModel::BGESmallENV15);
                    if let Some(dir) = cache_dir {
                        options = options.with_cache_dir(dir);
                    }
                    fastembed::TextEmbedding::try_new(options)
                })
                .await
                .map_err(|e| anyhow::anyhow!("embedding init task panicked: {e}"))? // join error
                .map_err(|e| anyhow::anyhow!("fastembed init failed: {e}"))?; // fastembed error
                Ok::<_, anyhow::Error>(Arc::new(std::sync::Mutex::new(init)))
            })
            .await
            .cloned()
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddings {
    async fn embed(&self, text: &str) -> anyhow::Result<EmbeddingResult> {
        let embedder = self.ensure_init().await?;
        let text = text.to_string();
        let vectors = tokio::task::spawn_blocking(move || {
            let mut guard = embedder
                .lock()
                .map_err(|e| anyhow::anyhow!("embedder mutex poisoned: {e}"))?;
            guard
                .embed(vec![text], None)
                .map_err(|e| anyhow::anyhow!("{e}"))
        })
        .await
        .map_err(|e| anyhow::anyhow!("embedding task panicked: {e}"))??;
        vectors
            .into_iter()
            .next()
            .map(|vector| EmbeddingResult {
                vector,
                model: self.model(),
            })
            .ok_or_else(|| anyhow::anyhow!("fastembed returned no vectors"))
    }

    fn model(&self) -> String {
        "bge-small-en-v1.5".to_string()
    }
}

/// Deterministic local embedding for unit tests.
/// Hashes text into a fixed-dimension vector. NOT semantically meaningful —
/// used only in unit tests where semantics don't matter.
#[allow(dead_code)]
pub struct HashEmbeddings {
    dims: usize,
}

#[allow(dead_code)]
impl HashEmbeddings {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

#[async_trait]
impl EmbeddingProvider for HashEmbeddings {
    async fn embed(&self, text: &str) -> anyhow::Result<EmbeddingResult> {
        // bag-of-character-trigrams hashed into dims buckets, L2-normalized
        let mut vector = vec![0f32; self.dims];
        let t = text.to_lowercase();
        for window in t.as_bytes().windows(3) {
            let digest = Md5::digest(window);
            let bucket = (((digest[0] as usize) << 8) | digest[1] as usize) % self.dims;
            vector[bucket] += 1.0;
        }
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        Ok(EmbeddingResult {
            vector,
            model: self.model(),
        })
    }

    fn model(&self) -> String {
        format!("local-hash-{}", self.dims)
    }
}

/// Configuration for provider selection (mirrors EmbeddingsConfig in @infra/config).
pub struct EmbeddingsConfig {
    pub provider: EmbeddingsProviderKind,
    pub baseurl: Option<String>,
    pub apikey: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingsProviderKind {
    Local,
    OpenAICompatible,
}

pub fn create_embedding_provider(config: &EmbeddingsConfig) -> Box<dyn EmbeddingProvider> {
    if config.provider == EmbeddingsProviderKind::OpenAICompatible
        && let (Some(baseurl), Some(model)) = (&config.baseurl, &config.model)
    {
        return Box::new(OpenAICompatibleEmbeddings::new(
            baseurl,
            config.apikey.as_deref().unwrap_or(""),
            model,
        ));
    }
    Box::new(LocalEmbeddings::new(local_cache_path()))
}

fn local_cache_path() -> Option<PathBuf> {
    std::env::var("FASTEMBED_CACHE_PATH").ok().map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::post;
    use axum::Json;
    use std::sync::Arc as StdArc;
    use tokio::sync::oneshot;

    #[tokio::test]
    async fn openai_compatible_posts_to_embeddings_with_model_and_bearer_header() {
        struct Captured {
            auth: Option<String>,
            body: serde_json::Value,
        }
        let (tx, rx) = oneshot::channel::<Captured>();
        let capture = StdArc::new(std::sync::Mutex::new(Some(tx)));

        let app = axum::Router::new().route(
            "/embeddings",
            post(move |headers: axum::http::HeaderMap, Json(body): Json<serde_json::Value>| {
                let capture = StdArc::clone(&capture);
                async move {
                    let _ = capture.lock().unwrap().take().unwrap().send(Captured {
                        auth: headers
                            .get(axum::http::header::AUTHORIZATION)
                            .and_then(|v| v.to_str().ok().map(String::from)),
                        body,
                    });
                    axum::Json(serde_json::json!({ "data": [{ "embedding": [0.1, 0.2] }] }))
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let provider =
            OpenAICompatibleEmbeddings::new(&format!("http://{addr}"), "sk-test", "emb-1");
        let result = provider.embed("hello").await.unwrap();
        assert_eq!(result.vector, vec![0.1, 0.2]);
        assert_eq!(result.model, "emb-1");

        let captured = rx.await.unwrap();
        assert_eq!(captured.body["model"], "emb-1");
        assert_eq!(captured.body["input"], "hello");
        assert_eq!(captured.auth.as_deref(), Some("Bearer sk-test"));
    }

    #[tokio::test]
    async fn no_apikey_means_no_authorization_header() {
        struct Captured {
            auth: Option<String>,
        }
        let (tx, rx) = oneshot::channel::<Captured>();
        let capture = StdArc::new(std::sync::Mutex::new(Some(tx)));

        let app = axum::Router::new().route(
            "/embeddings",
            post(move |headers: axum::http::HeaderMap| {
                let capture = StdArc::clone(&capture);
                async move {
                    let _ = capture.lock().unwrap().take().unwrap().send(Captured {
                        auth: headers
                            .get(axum::http::header::AUTHORIZATION)
                            .and_then(|v| v.to_str().ok().map(String::from)),
                    });
                    axum::Json(serde_json::json!({ "data": [{ "embedding": [1.0] }] }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let provider = OpenAICompatibleEmbeddings::new(&format!("http://{addr}"), "", "emb-1");
        provider.embed("hello").await.unwrap();
        assert!(rx.await.unwrap().auth.is_none());
    }

    #[tokio::test]
    async fn endpoint_error_surfaces_status_and_body() {
        let app = axum::Router::new().route(
            "/embeddings",
            post(|| async {
                (
                    axum::http::StatusCode::NOT_IMPLEMENTED,
                    axum::Json(serde_json::json!({ "error": "no embeddings support" })),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let provider = OpenAICompatibleEmbeddings::new(&format!("http://{addr}"), "", "emb-1");
        let err = format!("{:#}", provider.embed("hello").await.unwrap_err());
        assert!(err.contains("501"), "missing status: {err}");
        assert!(err.contains("no embeddings support"), "missing body: {err}");
    }

    #[test]
    fn create_embedding_provider_selects_openai_compatible_when_configured() {
        let p = create_embedding_provider(&EmbeddingsConfig {
            provider: EmbeddingsProviderKind::OpenAICompatible,
            baseurl: Some("http://x/v1".to_string()),
            apikey: None,
            model: Some("emb-1".to_string()),
        });
        assert_eq!(p.model(), "emb-1");
    }

    #[test]
    fn create_embedding_provider_falls_back_to_local_semantic_model() {
        let p = create_embedding_provider(&EmbeddingsConfig {
            provider: EmbeddingsProviderKind::Local,
            baseurl: None,
            apikey: None,
            model: None,
        });
        assert_eq!(p.model(), "bge-small-en-v1.5");
    }

    #[tokio::test]
    async fn hash_embeddings_deterministic_normalized_fixed_dims() {
        let h = HashEmbeddings::new(64);
        let a = h.embed("hello world").await.unwrap();
        let b = h.embed("hello world").await.unwrap();
        assert_eq!(a.vector, b.vector);
        assert_eq!(a.vector.len(), 64);
        let norm: f32 = a.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }
}
