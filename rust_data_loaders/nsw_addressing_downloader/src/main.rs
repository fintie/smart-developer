use anyhow::{anyhow, Context, Result};
use clap::Parser;
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, Clone, Copy)]
struct DatasetConfig {
    name: &'static str,
    base_url: &'static str,
    raw_subdir: &'static str,
    chunk_prefix: &'static str,
    chunk_size: usize,
    out_fields: &'static str,
    recursive_on_failure: bool,
}

const DATASETS: &[DatasetConfig] = &[
    DatasetConfig {
        name: "addressing",
        base_url: "https://portal.spatial.nsw.gov.au/server/rest/services/NSW_Geocoded_Addressing_Theme/FeatureServer/1/query",
        raw_subdir: "nsw_addressing",
        chunk_prefix: "addresspoint",
        chunk_size: 1000,
        out_fields: "*",
        recursive_on_failure: false,
    },
    DatasetConfig {
        name: "bushfire",
        base_url: "https://portal.spatial.nsw.gov.au/server/rest/services/Hosted/NSW_BushFire_Prone_Land/FeatureServer/0/query",
        raw_subdir: "nsw_bushfire",
        chunk_prefix: "bushfire",
        chunk_size: 1000,
        out_fields: "fid,d_category,d_guidelin,category,guideline,area",
        recursive_on_failure: true,
    },
    DatasetConfig {
        name: "flood",
        base_url: "https://mapprod3.environment.nsw.gov.au/arcgis/rest/services/Planning/Hazard/MapServer/1/query",
        raw_subdir: "nsw_flood",
        chunk_prefix: "flood",
        chunk_size: 500,
        out_fields: "OBJECTID,EPI_NAME,LGA_NAME,LAY_CLASS,EPI_TYPE,COMMENT",
        recursive_on_failure: true,
    },
    DatasetConfig {
        name: "heritage",
        base_url: "https://mapprod3.environment.nsw.gov.au/arcgis/rest/services/Planning/EPI_Primary_Planning_Layers/MapServer/0/query",
        raw_subdir: "nsw_heritage",
        chunk_prefix: "heritage",
        chunk_size: 1000,
        out_fields: "OBJECTID,EPI_NAME,LGA_NAME,LAY_CLASS,H_ID,H_NAME,SIG,LEGIS_REF_CLAUSE,PCO_REF_KEY,EPI_TYPE",
        recursive_on_failure: true,
    },
    DatasetConfig {
        name: "property",
        base_url: "https://portal.spatial.nsw.gov.au/server/rest/services/NSW_Land_Parcel_Property_Theme/FeatureServer/12/query",
        raw_subdir: "nsw_property",
        chunk_prefix: "property",
        chunk_size: 1000,
        out_fields: "RID,gurasid,principaladdresssiteoid,addressstringoid,propertytype,valnetpropertystatus,valnetpropertytype,dissolveparcelcount,valnetlotcount,propid,superlot,address,housenumber,urbanity,Shape__Area,Shape__Length",
        recursive_on_failure: true,
    },
    DatasetConfig {
        name: "zoning",
        base_url: "https://mapprod3.environment.nsw.gov.au/arcgis/rest/services/Planning/EPI_Primary_Planning_Layers/MapServer/2/query",
        raw_subdir: "nsw_zoning",
        chunk_prefix: "land_zoning",
        chunk_size: 500,
        out_fields: "OBJECTID,EPI_NAME,LGA_NAME,LAY_CLASS,SYM_CODE,PURPOSE,EPI_TYPE",
        recursive_on_failure: false,
    },
];

#[derive(Parser, Debug)]
#[command(name = "arcgis_downloader")]
struct Args {
    #[arg(long)]
    dataset: String,

    #[arg(long, default_value_t = 4)]
    max_concurrency: usize,

    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,

    #[arg(long, default_value_t = 50)]
    request_pause_ms: u64,

    #[arg(long)]
    sequential: bool,
}

#[derive(Debug, Deserialize)]
struct IdResponse {
    #[serde(rename = "objectIds")]
    object_ids: Option<Vec<i64>>,
}

fn get_dataset(name: &str) -> Result<DatasetConfig> {
    DATASETS
        .iter()
        .find(|d| d.name == name)
        .copied()
        .ok_or_else(|| anyhow!("unknown dataset: {name}"))
}

fn repo_root() -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("failed to get current working directory")?;
    let root = cwd
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow!("failed to infer repo root from current directory"))?;
    Ok(root.to_path_buf())
}

fn ensure_dirs(chunks_dir: &Path) -> Result<()> {
    fs::create_dir_all(chunks_dir)
        .with_context(|| format!("failed to create {}", chunks_dir.display()))?;
    Ok(())
}

fn chunk_vec(values: &[i64], chunk_size: usize) -> Vec<Vec<i64>> {
    values.chunks(chunk_size).map(|c| c.to_vec()).collect()
}

fn append_failed_id(failed_path: &Path, object_id: i64) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(failed_path)
        .with_context(|| format!("failed to open {}", failed_path.display()))?;

    writeln!(file, "{object_id}")
        .with_context(|| format!("failed to append to {}", failed_path.display()))?;
    Ok(())
}

async fn fetch_all_ids(client: &Client, cfg: &DatasetConfig, ids_path: &Path) -> Result<Vec<i64>> {
    let params = [
        ("where", "1=1"),
        ("returnIdsOnly", "true"),
        ("f", "json"),
    ];

    let response = client
        .get(cfg.base_url)
        .query(&params)
        .send()
        .await
        .with_context(|| format!("failed GET request for {}", cfg.name))?
        .error_for_status()
        .with_context(|| format!("non-success status when fetching IDs for {}", cfg.name))?;

    let text = response.text().await.context("failed to read ID response body")?;
    fs::write(ids_path, &text)
        .with_context(|| format!("failed to write {}", ids_path.display()))?;

    let parsed: IdResponse =
        serde_json::from_str(&text).context("failed to parse object ID response JSON")?;

    let ids = parsed
        .object_ids
        .ok_or_else(|| anyhow!("response did not contain objectIds"))?;

    if ids.is_empty() {
        return Err(anyhow!("No objectIds returned. Check URL or layer access."));
    }

    Ok(ids)
}

async fn fetch_chunk_once(
    client: &Client,
    cfg: &DatasetConfig,
    object_ids: &[i64],
    out_path: &Path,
    pause_ms: u64,
) -> Result<()> {
    if pause_ms > 0 {
        sleep(Duration::from_millis(pause_ms)).await;
    }

    let object_ids_str = object_ids
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let payload = [
        ("objectIds", object_ids_str),
        ("outFields", cfg.out_fields.to_string()),
        ("returnGeometry", "true".to_string()),
        ("f", "geojson".to_string()),
    ];

    let response = client
        .post(cfg.base_url)
        .form(&payload)
        .send()
        .await
        .with_context(|| format!("failed POST request for {}", out_path.display()))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .with_context(|| format!("failed reading body for {}", out_path.display()))?;

    if !status.is_success() {
        let preview: String = text.chars().take(1000).collect();
        return Err(anyhow!("status {}: {}", status, preview));
    }

    fs::write(out_path, text)
        .with_context(|| format!("failed writing {}", out_path.display()))?;

    Ok(())
}

async fn fetch_chunk_recursive(
    client: &Client,
    cfg: &DatasetConfig,
    object_ids: &[i64],
    chunk_name: &str,
    chunks_dir: &Path,
    failed_ids_path: &Path,
    pause_ms: u64,
) -> Result<()> {
    let out_path = chunks_dir.join(format!("{chunk_name}.geojson"));

    if out_path.exists() {
        return Ok(());
    }

    let result = fetch_chunk_once(client, cfg, object_ids, &out_path, pause_ms).await;

    match result {
        Ok(_) => {
            println!(
                "Saved {} ({} ids)",
                out_path.file_name().unwrap().to_string_lossy(),
                object_ids.len()
            );
            Ok(())
        }
        Err(e) => {
            println!(
                "Failed chunk {} with {} ids: {}",
                chunk_name,
                object_ids.len(),
                e
            );

            if object_ids.len() == 1 {
                let bad_id = object_ids[0];
                println!("Single OBJECTID failed permanently: {}", bad_id);
                append_failed_id(failed_ids_path, bad_id)?;
                return Ok(());
            }

            let mid = object_ids.len() / 2;
            let left = &object_ids[..mid];
            let right = &object_ids[mid..];

            let left_name = format!("{chunk_name}_a");
            let right_name = format!("{chunk_name}_b");

            Box::pin(fetch_chunk_recursive(
                client,
                cfg,
                left,
                &left_name,
                chunks_dir,
                failed_ids_path,
                pause_ms,
            ))
            .await?;

            Box::pin(fetch_chunk_recursive(
                client,
                cfg,
                right,
                &right_name,
                chunks_dir,
                failed_ids_path,
                pause_ms,
            ))
            .await?;

            Ok(())
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = get_dataset(&args.dataset)?;

    let root = repo_root()?;
    let raw_dir = root.join("data").join("raw").join(cfg.raw_subdir);
    let chunks_dir = raw_dir.join("chunks");
    let ids_path = raw_dir.join("ids.json");
    let failed_ids_path = chunks_dir.join("failed_ids.txt");
    let failed_chunks_path = raw_dir.join("failed_chunks.json");

    ensure_dirs(&chunks_dir)?;

    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout_secs))
        .pool_idle_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(30))
        .user_agent("arcgis-downloader-rust/0.1")
        .build()
        .context("failed to build reqwest client")?;

    println!("Dataset: {}", cfg.name);
    println!("Fetching all object IDs...");
    let object_ids = fetch_all_ids(&client, &cfg, &ids_path).await?;
    println!("Total object IDs: {}", object_ids.len());

    let chunks = chunk_vec(&object_ids, cfg.chunk_size);
    let total_chunks = chunks.len();
    println!("Prepared {} chunks.", total_chunks);

    if cfg.recursive_on_failure {
        println!("Mode: recursive split on failure");

        for (idx0, chunk_ids) in chunks.into_iter().enumerate() {
            let idx = idx0 + 1;
            let chunk_name = format!("{}_{:04}_of_{:04}", cfg.chunk_prefix, idx, total_chunks);
            let out_path = chunks_dir.join(format!("{chunk_name}.geojson"));

            if out_path.exists() {
                if idx <= 5 || idx % 100 == 0 || idx == total_chunks {
                    println!(
                        "[{}/{}] Skipping existing chunk: {}",
                        idx,
                        total_chunks,
                        out_path.file_name().unwrap().to_string_lossy()
                    );
                }
                continue;
            }

            if idx <= 5 || idx % 100 == 0 || idx == total_chunks {
                println!("[{}/{}] Processing {} ids...", idx, total_chunks, chunk_ids.len());
            }

            fetch_chunk_recursive(
                &client,
                &cfg,
                &chunk_ids,
                &chunk_name,
                &chunks_dir,
                &failed_ids_path,
                args.request_pause_ms,
            )
            .await?;

            if idx % 100 == 0 || idx == total_chunks {
                println!("Progress: completed {} / {} top-level chunks", idx, total_chunks);
            }
        }
    } else {
        println!("Mode: concurrent chunk download");

        let mut todo: Vec<(usize, Vec<i64>, PathBuf)> = Vec::new();
        for (idx0, chunk_ids) in chunks.into_iter().enumerate() {
            let idx = idx0 + 1;
            let out_path = chunks_dir.join(format!(
                "{}_{:04}_of_{:04}.geojson",
                cfg.chunk_prefix, idx, total_chunks
            ));

            if out_path.exists() {
                continue;
            }

            todo.push((idx, chunk_ids, out_path));
        }

        println!("Chunks to download: {}", todo.len());
        if todo.is_empty() {
            println!("Done. All chunks already exist.");
            return Ok(());
        }

        let results = if args.sequential {
            let mut out = Vec::new();
            for (idx, chunk_ids, out_path) in todo {
                let result =
                    fetch_chunk_once(&client, &cfg, &chunk_ids, &out_path, args.request_pause_ms)
                        .await;
                out.push((idx, out_path, result));
            }
            out
        } else {
            stream::iter(todo.into_iter().map(|(idx, chunk_ids, out_path)| {
                let client = client.clone();
                let cfg = cfg;
                let pause_ms = args.request_pause_ms;
                async move {
                    let result =
                        fetch_chunk_once(&client, &cfg, &chunk_ids, &out_path, pause_ms).await;
                    (idx, out_path, result)
                }
            }))
            .buffer_unordered(args.max_concurrency)
            .collect::<Vec<_>>()
            .await
        };

        let mut failed_chunks: Vec<usize> = Vec::new();
        let mut completed = 0usize;
        let total_to_download = results.len();

        for (idx, out_path, result) in results {
            completed += 1;

            match result {
                Ok(_) => {
                    if completed <= 5 || completed % 100 == 0 || completed == total_to_download {
                        println!(
                            "[{}/{}] Downloaded chunk {}: {}",
                            completed,
                            total_to_download,
                            idx,
                            out_path.file_name().unwrap().to_string_lossy()
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[{}/{}] Failed chunk {}: {}",
                        completed,
                        total_to_download,
                        idx,
                        e
                    );
                    failed_chunks.push(idx);
                }
            }

            if completed % 100 == 0 || completed == total_to_download {
                println!(
                    "Progress: completed {} / {} chunks, failed so far: {}",
                    completed,
                    total_to_download,
                    failed_chunks.len()
                );
            }
        }

        if !failed_chunks.is_empty() {
            failed_chunks.sort_unstable();
            let payload = json!({ "failed_chunks": failed_chunks });
            fs::write(&failed_chunks_path, serde_json::to_string_pretty(&payload)?)
                .with_context(|| format!("failed to write {}", failed_chunks_path.display()))?;
            println!(
                "Finished with failures. Failed chunk indices saved to: {}",
                failed_chunks_path.display()
            );
        }
    }

    println!("Done.");
    Ok(())
}