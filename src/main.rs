use actix_cors::Cors;
use actix_files as fs;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use anyhow::{anyhow, Result};
use local_ip_address::local_ip;
use opencv::{
    prelude::*, videoio, imgcodecs, imgproc, dnn,
    core::{Vector, Size, Mat, Scalar, Ptr},
    objdetect::FaceDetectorYN,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs as std_fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::task;

// --- CONFIGURATION ---
const CONFIG_FILE: &str = "config.json";
const STORAGE_FOLDER: &str = "captures";
const QUEUE_SIZE: usize = 100;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct SystemConfig {
    system_models: HashMap<String, String>,
    cameras: Vec<CameraConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct CameraConfig {
    id: usize,
    name: String,
    url: String,
}

// --- HYBRID ENGINE (Supports both Face & YOLO) ---
enum AiModel {
    Face(Ptr<FaceDetectorYN>),
    Yolo(dnn::Net),
}

// --- APP STATE (Shared Monitor) ---
struct AppState {
    active_switches: Mutex<HashMap<usize, HashSet<String>>>, 
    config: SystemConfig,
    // Monitor Stats
    processed_count: Mutex<u64>,
    skipped_count: Mutex<u64>,
    captured_count: Mutex<u64>,
}

#[derive(Deserialize)]
struct ToggleReq {
    cam_id: usize,
    model: String,
    enabled: bool,
}

// --- API ---
#[get("/api/init")]
async fn api_init(data: web::Data<Arc<AppState>>) -> impl Responder {
    HttpResponse::Ok().json(&data.config)
}

#[get("/api/status/{cam_id}")]
async fn api_status(data: web::Data<Arc<AppState>>, path: web::Path<usize>) -> impl Responder {
    let cam_id = path.into_inner();
    let switches = data.active_switches.lock().unwrap();
    let active: Vec<String> = switches.get(&cam_id)
        .map(|s| s.iter().cloned().collect())
        .unwrap_or_default();
    HttpResponse::Ok().json(active)
}

#[post("/api/toggle")]
async fn api_toggle(data: web::Data<Arc<AppState>>, req: web::Json<ToggleReq>) -> impl Responder {
    let mut switches = data.active_switches.lock().unwrap();
    let cam_set = switches.entry(req.cam_id).or_insert_with(HashSet::new);
    if req.enabled {
        cam_set.insert(req.model.clone());
        println!("🟢 ENABLED [{}] on Camera {}", req.model, req.cam_id);
    } else {
        cam_set.remove(&req.model);
        println!("🔴 DISABLED [{}] on Camera {}", req.model, req.cam_id);
    }
    HttpResponse::Ok().body("Updated")
}

#[get("/api/events")]
async fn api_events() -> impl Responder {
    let mut files = Vec::new();
    if let Ok(entries) = std_fs::read_dir(STORAGE_FOLDER) {
        for entry in entries.flatten() {
            if let Ok(name) = entry.file_name().into_string() {
                if name.ends_with(".jpg") { files.push(name); }
            }
        }
    }
    HttpResponse::Ok().json(files)
}

#[tokio::main]
async fn main() -> Result<()> {
    std::env::set_var("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp");
    let running = Arc::new(AtomicBool::new(true));

    for dir in [STORAGE_FOLDER, "static", "models"] {
        if !Path::new(dir).exists() { std_fs::create_dir(dir)?; }
    }

    // Load Config
    let config_raw = std_fs::read_to_string(CONFIG_FILE).expect("Failed to read config.json");
    let config: SystemConfig = serde_json::from_str(&config_raw).expect("Invalid JSON");

    let app_state = Arc::new(AppState {
        active_switches: Mutex::new(HashMap::new()),
        config: config.clone(),
        processed_count: Mutex::new(0),
        skipped_count: Mutex::new(0),
        captured_count: Mutex::new(0),
    });

    println!("======================================================");
    println!(" 🚀 AI SYSTEM ONLINE: http://{}:8080", local_ip().unwrap_or([127,0,0,1].into()));
    println!("======================================================");

    let (tx, rx) = flume::bounded::<(usize, Vec<u8>)>(QUEUE_SIZE);

    // 1. Web Server
    let state_ref = app_state.clone();
    let srv = HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state_ref.clone()))
            .wrap(Cors::permissive())
            .service(api_init)
            .service(api_status)
            .service(api_toggle)
            .service(api_events)
            .service(fs::Files::new("/captures", STORAGE_FOLDER))
            .service(fs::Files::new("/", "static").index_file("index.html"))
    })
    .bind(("0.0.0.0", 8080))?
    .run();
    let srv_handle = tokio::spawn(srv);

    // 2. Camera Producers
    for cam in config.cameras.clone() {
        let p_tx = tx.clone();
        let p_running = running.clone();
        let p_state = app_state.clone();
        task::spawn_blocking(move || {
            while p_running.load(Ordering::SeqCst) {
                if let Err(_) = run_producer(cam.clone(), p_tx.clone(), &p_running, &p_state) {}
                std::thread::sleep(Duration::from_secs(5));
            }
        });
    }

    // 3. AI Consumer
    let c_state = app_state.clone();
    let cam_names: HashMap<usize, String> = config.cameras.iter().map(|c| (c.id, c.name.clone())).collect();
    let c_running = running.clone();

    task::spawn_blocking(move || {
        // Load Models (Soft Fail)
        let mut models: HashMap<String, AiModel> = HashMap::new();
        
        for (key, path) in &c_state.config.system_models {
            if key == "face" {
                match FaceDetectorYN::create(path, "", Size::new(0, 0), 0.6, 0.3, 5000, 0, 0) {
                    Ok(m) => { 
                        println!("✅ LOADED: [Face] {}", key);
                        models.insert(key.clone(), AiModel::Face(m));
                    },
                    Err(_) => eprintln!("⚠️ SKIPPED: [Face] Model missing at {}", path),
                }
            } else {
                match dnn::read_net_from_onnx(path) {
                    Ok(mut net) => {
                        let _ = net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV);
                        let _ = net.set_preferable_target(dnn::DNN_TARGET_CPU);
                        println!("✅ LOADED: [YOLO] {}", key);
                        models.insert(key.clone(), AiModel::Yolo(net));
                    },
                    Err(_) => eprintln!("⚠️ SKIPPED: [YOLO] Model missing at {}", path),
                }
            }
        }

        while c_running.load(Ordering::SeqCst) {
            if let Ok((cam_id, img_data)) = rx.recv_timeout(Duration::from_millis(500)) {
                let cam_name = cam_names.get(&cam_id).unwrap();
                
                // Check switches
                let active_models: Vec<String> = {
                    let lock = c_state.active_switches.lock().unwrap();
                    lock.get(&cam_id).cloned().unwrap_or_default().into_iter().collect()
                };

                let mut processed = false;
                if !active_models.is_empty() {
                    if let Ok(frame) = imgcodecs::imdecode(&Vector::from_slice(&img_data), imgcodecs::IMREAD_COLOR) {
                        for model_key in active_models {
                            if let Some(ai_model) = models.get_mut(&model_key) {
                                match ai_model {
                                    AiModel::Face(detector) => {
                                        let _ = run_face_inference(&frame, detector, &model_key, cam_name);
                                    },
                                    AiModel::Yolo(net) => {
                                        let _ = run_yolo_inference(&frame, net, &model_key, cam_name);
                                    }
                                }
                                processed = true;
                            }
                        }
                    }
                }
                
                // Update Stats
                if processed {
                    *c_state.processed_count.lock().unwrap() += 1;
                } else {
                    *c_state.skipped_count.lock().unwrap() += 1;
                }
            }
        }
    });

    // 4. Monitor Loop
    let m_state = app_state.clone();
    let m_running = running.clone();
    tokio::spawn(async move {
        while m_running.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let p = *m_state.processed_count.lock().unwrap();
            let s = *m_state.skipped_count.lock().unwrap();
            let c = *m_state.captured_count.lock().unwrap();
            println!("[SYSTEM] Captured: {} | Processed: {} | Skipped: {}", c, p, s);
        }
    });

    // 5. Ctrl+C Shutdown
    tokio::signal::ctrl_c().await?;
    println!("\n🛑 Stopping System...");
    running.store(false, Ordering::SeqCst);
    srv_handle.abort(); // Kill web server
    std::thread::sleep(Duration::from_secs(1)); // Allow threads to exit
    std::process::exit(0);
}

fn run_producer(conf: CameraConfig, tx: flume::Sender<(usize, Vec<u8>)>, running: &Arc<AtomicBool>, state: &Arc<AppState>) -> Result<()> {
    let mut cam = videoio::VideoCapture::from_file(&conf.url, videoio::CAP_FFMPEG)?;
    if !videoio::VideoCapture::is_opened(&cam)? { return Err(anyhow!("Fail")); }
    let mut frame = Mat::default();
    let mut last_send = Instant::now();

    while running.load(Ordering::SeqCst) {
        if !cam.read(&mut frame)? { break; }
        if frame.empty() { continue; }
        
        // 2 FPS Limit
        if last_send.elapsed() >= Duration::from_millis(500) {
            let mut buf = Vector::new();
            imgcodecs::imencode(".jpg", &frame, &mut buf, &Vector::new())?;
            if tx.send((conf.id, buf.to_vec())).is_ok() {
                *state.captured_count.lock().unwrap() += 1;
            }
            last_send = Instant::now();
        }
    }
    Ok(())
}

fn run_face_inference(frame: &Mat, detector: &mut Ptr<FaceDetectorYN>, model_name: &str, cam_name: &str) -> Result<()> {
    let mut small = Mat::default();
    let size = Size::new(640, 360);
    imgproc::resize(frame, &mut small, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;
    detector.set_input_size(size)?;
    let mut faces = Mat::default();
    detector.detect(&small, &mut faces)?;

    if faces.rows() > 0 {
        // FORCE UPPERCASE NAMING: Cam_FACE_Time.jpg
        let fname = format!("{}/{}_{}_{}.jpg", STORAGE_FOLDER, cam_name, model_name.to_uppercase(), chrono::Local::now().format("%H-%M-%S"));
        imgcodecs::imwrite(&fname, frame, &Vector::new())?;
        println!("🚨 DETECTED: {} on {}", model_name, cam_name);
    }
    Ok(())
}

fn run_yolo_inference(frame: &Mat, net: &mut dnn::Net, model_name: &str, cam_name: &str) -> Result<()> {
    let blob = dnn::blob_from_image(frame, 1.0/255.0, Size::new(640, 640), Scalar::default(), true, false, 32)?;
    net.set_input(&blob, "", 1.0, Scalar::default())?;
    
    let mut outputs = Vector::<Mat>::new();
    let out_names = net.get_unconnected_out_layers_names()?;
    net.forward(&mut outputs, &out_names)?;

    let output = outputs.get(0)?;
    if output.size()?.height > 0 {
         // FORCE UPPERCASE NAMING
         let fname = format!("{}/{}_{}_{}.jpg", STORAGE_FOLDER, cam_name, model_name.to_uppercase(), chrono::Local::now().format("%H-%M-%S"));
         imgcodecs::imwrite(&fname, frame, &Vector::new())?;
         println!("🚨 DETECTED: {} on {}", model_name, cam_name);
    }
    Ok(())
}