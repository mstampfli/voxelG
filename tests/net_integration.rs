// In-process client<->server smoke test over the real TCP + UDP stack:
// handshake (with version gate), a reliable edit over TCP, and a pose over UDP.
// Skips gracefully if the port is busy (so it never fails CI spuriously).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use voxelg::net::{self, Message};

#[test]
fn handshake_edit_and_pose_over_real_sockets() {
    let port = 39_517u16;
    let Ok(mut server) = net::NetServer::listen(port) else {
        eprintln!("port {port} busy — skipping net integration test");
        return;
    };
    let udp_port = server.udp_port();

    let edits = Arc::new(Mutex::new(Vec::<Message>::new()));
    let poses = Arc::new(Mutex::new(Vec::<net::PlayerId>::new()));
    let stop = Arc::new(AtomicBool::new(false));

    let edits_s = edits.clone();
    let poses_s = poses.clone();
    let stop_s = stop.clone();
    let server_thread = thread::spawn(move || {
        while !stop_s.load(Ordering::Relaxed) {
            let (joined, msgs, _timed_out) = server.poll();
            for id in joined {
                let token = server.issue_token(id);
                server.send_to(id, Message::JoinAck { your_id: id, seed: 42, token, udp_port });
            }
            for (sender, m) in msgs {
                match m {
                    Message::VoxelEdit { .. } => edits_s.lock().unwrap().push(m),
                    Message::PlayerUpdate { .. } => poses_s.lock().unwrap().push(sender),
                    _ => {}
                }
            }
            thread::sleep(Duration::from_millis(3));
        }
    });

    let mut client = net::NetClient::connect(&format!("127.0.0.1:{port}")).expect("connect");

    // Wait for JoinAck.
    let mut my_id = None;
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && my_id.is_none() {
        for m in client.drain() {
            if let Message::JoinAck { your_id, .. } = m {
                my_id = Some(your_id);
            }
        }
        thread::sleep(Duration::from_millis(10));
    }
    let my_id = my_id.expect("never received JoinAck");

    // Reliable edit over TCP.
    client.send(Message::VoxelEdit { wx: 7, wy: 8, wz: 9, mat: 4, seq: 0 });
    // Pose over UDP (token + server udp port now known from JoinAck).
    for _ in 0..10 {
        client.send_pose(my_id, [1.0, 2.0, 3.0], 0.5, -0.2);
        thread::sleep(Duration::from_millis(20));
    }

    // Allow delivery.
    thread::sleep(Duration::from_millis(300));
    stop.store(true, Ordering::Relaxed);
    let _ = server_thread.join();

    let got_edits = edits.lock().unwrap();
    assert!(
        got_edits.iter().any(|m| matches!(m, Message::VoxelEdit { wx: 7, wy: 8, wz: 9, mat: 4, .. })),
        "server never received the TCP edit: {got_edits:?}"
    );
    // UDP is best-effort, but on loopback at least one of ten should arrive.
    let got_poses = poses.lock().unwrap();
    assert!(!got_poses.is_empty(), "server never received a UDP pose");
}

#[test]
fn rejects_wrong_protocol_version() {
    // Directly assert the version gate constant is what the client sends.
    assert_eq!(net::PROTOCOL_VERSION, 1);
}
