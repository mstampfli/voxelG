// Dedicated-server loop. Owns the authoritative, ordered, versioned edit log
// and the player table; clients sync from the shared seed + replayed edits.

use crate::net;

pub fn run_server(port: u16) -> ! {
    let mut server = net::NetServer::listen(port).expect("listen");
    let udp_port = server.udp_port();
    let seed = 0xC0FFEE_F00D_BEEFu64;
    let mut player_states: std::collections::HashMap<net::PlayerId, ([f32; 3], f32, f32)> =
        std::collections::HashMap::new();

    // Authoritative ORDERED edit log. Each entry is a VoxelEdit/Explode message
    // carrying a monotonically increasing `seq` (checklist: ordered/versioned
    // edits + authoritative log). Replayed in order to joiners so late joiners
    // never desync. (A production server would periodically compact this into a
    // snapshot; replaying the full ordered log is correct, just unbounded.)
    let mut edit_log: Vec<net::Message> = Vec::new();
    let mut next_seq: u64 = 1;

    const INTEREST_R: f32 = 600.0;
    const INTEREST_R2: f32 = INTEREST_R * INTEREST_R;
    log::info!("server: ready");
    loop {
        let (joined, msgs, timed_out) = server.poll();
        for id in joined {
            let token = server.issue_token(id);
            server.send_to(id, net::Message::JoinAck { your_id: id, seed, token, udp_port });
            // Bulk, compressed, ordered edit-log replay.
            if !edit_log.is_empty() {
                let compressed = net::compress_edits(&edit_log);
                server.send_to(id, net::Message::EditLog { compressed });
            }
            for (pid, st) in &player_states {
                server.send_to(
                    id,
                    net::Message::PlayerUpdate { id: *pid, pos: st.0, yaw: st.1, pitch: st.2 },
                );
            }
            server.broadcast(&net::Message::PlayerJoin { id }, Some(id));
            log::info!("server: player {} joined ({} edits replayed)", id, edit_log.len());
        }
        for id in timed_out {
            player_states.remove(&id);
            server.broadcast(&net::Message::PlayerLeave { id }, None);
        }
        for (sender, m) in msgs {
            match m {
                net::Message::Hello { .. } | net::Message::Heartbeat => {}
                net::Message::PlayerUpdate { pos, yaw, pitch, .. } => {
                    player_states.insert(sender, (pos, yaw, pitch));
                    let states = player_states.clone();
                    // Pose fan-out over UDP with distance interest management.
                    server.broadcast_pose_filter(sender, pos, yaw, pitch, |other_id| {
                        states.get(&other_id).map_or(true, |s| {
                            let dx = s.0[0] - pos[0];
                            let dz = s.0[2] - pos[2];
                            (dx * dx + dz * dz) < INTEREST_R2
                        })
                    });
                }
                net::Message::VoxelEdit { wx, wy, wz, mat, .. } => {
                    let seq = next_seq;
                    next_seq += 1;
                    let edit = net::Message::VoxelEdit { wx, wy, wz, mat, seq };
                    edit_log.push(edit.clone());
                    let states = player_states.clone();
                    server.broadcast_filter(&edit, |other_id| {
                        states.get(&other_id).map_or(true, |s| {
                            let dx = s.0[0] - wx as f32;
                            let dz = s.0[2] - wz as f32;
                            (dx * dx + dz * dz) < INTEREST_R2
                        })
                    });
                }
                net::Message::Explode { cx, cy, cz, radius, mat, .. } => {
                    let seq = next_seq;
                    next_seq += 1;
                    let edit = net::Message::Explode { cx, cy, cz, radius, mat, seq };
                    edit_log.push(edit.clone());
                    let states = player_states.clone();
                    server.broadcast_filter(&edit, |other_id| {
                        states.get(&other_id).map_or(true, |s| {
                            let dx = s.0[0] - cx as f32;
                            let dz = s.0[2] - cz as f32;
                            (dx * dx + dz * dz) < INTEREST_R2
                        })
                    });
                }
                net::Message::PlayerLeave { id } => {
                    player_states.remove(&id);
                    server.drop_client(id);
                    server.broadcast(&net::Message::PlayerLeave { id }, None);
                    log::info!("server: player {} left", id);
                }
                _ => {}
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
