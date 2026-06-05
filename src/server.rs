// Dedicated-server loop. Split out of main.rs (checklist: hygiene). Owns the
// authoritative player table and the persistent edit log; clients sync world
// state from the shared seed + replayed edits.

use crate::net;

pub fn run_server(port: u16) -> ! {
    let mut server = net::NetServer::listen(port).expect("listen");
    let seed = 0xC0FFEE_F00D_BEEFu64;
    let mut player_states: std::collections::HashMap<net::PlayerId, ([f32; 3], f32, f32)> =
        std::collections::HashMap::new();
    // Persistent edit log on the server. Each entry overrides the seed-noise
    // generation when a chunk loads. Sent in full to every new client and
    // appended on every VoxelEdit received.
    let mut edits: std::collections::HashMap<(i32, i32, i32), u8> =
        std::collections::HashMap::new();
    const INTEREST_R: f32 = 600.0;
    const INTEREST_R2: f32 = INTEREST_R * INTEREST_R;
    log::info!("server: ready");
    loop {
        let (joined, msgs) = server.poll();
        for id in joined {
            server.send_to(id, net::Message::JoinAck { your_id: id, seed });
            for (&(wx, wy, wz), &mat) in &edits {
                server.send_to(id, net::Message::VoxelEdit { wx, wy, wz, mat });
            }
            for (pid, st) in &player_states {
                server.send_to(
                    id,
                    net::Message::PlayerUpdate {
                        id: *pid,
                        pos: st.0,
                        yaw: st.1,
                        pitch: st.2,
                    },
                );
            }
            server.broadcast(&net::Message::PlayerJoin { id }, Some(id));
            log::info!("server: player {} joined ({} edits replayed)", id, edits.len());
        }
        for (sender, m) in msgs {
            match m {
                net::Message::Hello => {}
                net::Message::PlayerUpdate { pos, yaw, pitch, .. } => {
                    player_states.insert(sender, (pos, yaw, pitch));
                    let states = player_states.clone();
                    server.broadcast_filter(
                        &net::Message::PlayerUpdate { id: sender, pos, yaw, pitch },
                        |other_id| {
                            if other_id == sender { return false; }
                            states.get(&other_id).map_or(true, |s| {
                                let dx = s.0[0] - pos[0];
                                let dz = s.0[2] - pos[2];
                                (dx * dx + dz * dz) < INTEREST_R2
                            })
                        },
                    );
                }
                net::Message::VoxelEdit { wx, wy, wz, mat } => {
                    edits.insert((wx, wy, wz), mat);
                    let states = player_states.clone();
                    server.broadcast_filter(
                        &net::Message::VoxelEdit { wx, wy, wz, mat },
                        |other_id| {
                            states.get(&other_id).map_or(true, |s| {
                                let dx = s.0[0] - wx as f32;
                                let dz = s.0[2] - wz as f32;
                                (dx * dx + dz * dz) < INTEREST_R2
                            })
                        },
                    );
                }
                net::Message::Explode { cx, cy, cz, radius, mat } => {
                    let r = radius as i32;
                    let r2 = r * r;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            for dz in -r..=r {
                                if dx * dx + dy * dy + dz * dz > r2 { continue; }
                                edits.insert((cx + dx, cy + dy, cz + dz), mat);
                            }
                        }
                    }
                    let states = player_states.clone();
                    server.broadcast_filter(
                        &net::Message::Explode { cx, cy, cz, radius, mat },
                        |other_id| {
                            states.get(&other_id).map_or(true, |s| {
                                let dx = s.0[0] - cx as f32;
                                let dz = s.0[2] - cz as f32;
                                (dx * dx + dz * dz) < INTEREST_R2
                            })
                        },
                    );
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
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
