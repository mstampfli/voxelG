// Thin launcher: parse the run mode and dispatch into the library. All engine
// code lives in the `voxelg` library crate (src/lib.rs); see app.rs (client),
// server.rs (dedicated server) and voxel.rs (world).

use voxelg::{app, net, server};

enum Mode {
    Solo,
    Server(u16),
    Connect(String),
}

fn parse_mode() -> Mode {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--server" => {
                let port = args
                    .get(i + 1)
                    .and_then(|s| s.parse::<u16>().ok())
                    .unwrap_or(7878);
                return Mode::Server(port);
            }
            "--connect" => {
                let addr = args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| "127.0.0.1:7878".to_string());
                return Mode::Connect(addr);
            }
            _ => {}
        }
        i += 1;
    }
    Mode::Solo
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    match parse_mode() {
        Mode::Server(port) => server::run_server(port),
        mode => {
            let net = match mode {
                Mode::Connect(addr) => match net::NetClient::connect(&addr) {
                    Ok(c) => {
                        log::info!("connected to {}", addr);
                        Some(c)
                    }
                    Err(e) => {
                        log::error!("connect failed: {}", e);
                        None
                    }
                },
                _ => None,
            };
            app::run_client(net);
        }
    }
}
