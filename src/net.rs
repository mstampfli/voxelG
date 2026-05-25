// Lock-free multiplayer.
//
// Architecture:
// * Each TCP connection gets two threads: a reader that pushes inbound
//   messages into a crossbeam channel, and a writer that pulls outbound
//   messages from another crossbeam channel and writes them to the socket.
// * The main game-loop thread polls the inbound channel and pushes to the
//   outbound channels. Mutable state (player table, world) is owned by that
//   single thread — no shared mutexes anywhere.
// * Crossbeam's MPMC channels are lock-free (compare-exchange on atomics).
//
// World sync is via shared seed: every client deterministically regenerates
// the same terrain locally. Network only carries player state + voxel edits.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError};
use serde::{Deserialize, Serialize};

pub type PlayerId = u32;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    /// Client → server on connect.
    Hello,
    /// Server → new client: hands out the player id and the world seed.
    JoinAck { your_id: PlayerId, seed: u64 },
    /// Sent both ways: client tells server its current pose, server fans
    /// out to other clients.
    PlayerUpdate {
        id: PlayerId,
        pos: [f32; 3],
        yaw: f32,
        pitch: f32,
    },
    /// Server → other clients when someone joins / leaves.
    PlayerJoin { id: PlayerId },
    PlayerLeave { id: PlayerId },
    /// Voxel edit: world voxel coord + new material. Broadcast to everyone.
    VoxelEdit { wx: i32, wy: i32, wz: i32, mat: u8 },
    /// Sphere-of-impact: replace every voxel within `radius` of the centre
    /// with `mat`. One message instead of O(r³) VoxelEdits.
    Explode { cx: i32, cy: i32, cz: i32, radius: u8, mat: u8 },
}

fn read_message<R: Read>(r: &mut R) -> std::io::Result<Message> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 4 * 1024 * 1024 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "msg too big"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    bincode::deserialize(&buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn write_message<W: Write>(w: &mut W, msg: &Message) -> std::io::Result<()> {
    let buf = bincode::serialize(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    w.write_all(&(buf.len() as u32).to_le_bytes())?;
    w.write_all(&buf)?;
    w.flush()
}

// ---------------- client ----------------

pub struct NetClient {
    out: Sender<Message>,
    inbox: Receiver<Message>,
    /// Cached player id from JoinAck; main loop reads it once.
    pub my_id: Option<PlayerId>,
    pub seed: Option<u64>,
}

impl NetClient {
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        let stream_r = stream.try_clone()?;
        let stream_w = stream;
        let (tx_inbound, rx_inbound) = unbounded::<Message>();
        let (tx_outbound, rx_outbound) = unbounded::<Message>();

        thread::spawn(move || {
            let mut r = std::io::BufReader::new(stream_r);
            loop {
                match read_message(&mut r) {
                    Ok(m) => {
                        if tx_inbound.send(m).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        thread::spawn(move || {
            let mut w = std::io::BufWriter::new(stream_w);
            for m in rx_outbound.iter() {
                if write_message(&mut w, &m).is_err() {
                    break;
                }
            }
        });

        let client = NetClient {
            out: tx_outbound,
            inbox: rx_inbound,
            my_id: None,
            seed: None,
        };
        // Kick off with Hello.
        let _ = client.out.send(Message::Hello);
        Ok(client)
    }

    pub fn send(&self, msg: Message) {
        let _ = self.out.send(msg);
    }

    /// Drain all currently-buffered messages.
    pub fn drain(&mut self) -> Vec<Message> {
        let mut out = Vec::new();
        loop {
            match self.inbox.try_recv() {
                Ok(m) => {
                    if let Message::JoinAck { your_id, seed } = &m {
                        self.my_id = Some(*your_id);
                        self.seed = Some(*seed);
                    }
                    out.push(m);
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return out,
            }
        }
    }
}

// ---------------- server ----------------

struct NewClient {
    id: PlayerId,
    tx: Sender<Message>,
}

pub struct NetServer {
    inbound: Receiver<(PlayerId, Message)>,
    new_clients: Receiver<NewClient>,
    /// Owner: server's main thread.
    clients: HashMap<PlayerId, Sender<Message>>,
}

impl NetServer {
    pub fn listen(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        log::info!("server listening on 0.0.0.0:{}", port);
        let next_id = Arc::new(AtomicU32::new(1));
        let (tx_inbound, rx_inbound) = unbounded::<(PlayerId, Message)>();
        let (tx_new, rx_new) = unbounded::<NewClient>();

        thread::spawn(move || {
            for stream_res in listener.incoming() {
                let Ok(stream) = stream_res else { continue; };
                let id = next_id.fetch_add(1, Ordering::Relaxed);
                let _ = stream.set_nodelay(true);
                let Ok(stream_r) = stream.try_clone() else { continue; };
                let stream_w = stream;
                let (tx_out, rx_out) = unbounded::<Message>();

                let tx_in_clone = tx_inbound.clone();
                thread::spawn(move || {
                    let mut r = std::io::BufReader::new(stream_r);
                    loop {
                        match read_message(&mut r) {
                            Ok(m) => {
                                if tx_in_clone.send((id, m)).is_err() {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    // Signal disconnect via PlayerLeave from the reader side.
                    let _ = tx_in_clone.send((id, Message::PlayerLeave { id }));
                });
                thread::spawn(move || {
                    let mut w = std::io::BufWriter::new(stream_w);
                    for m in rx_out.iter() {
                        if write_message(&mut w, &m).is_err() {
                            break;
                        }
                    }
                });
                let _ = tx_new.send(NewClient { id, tx: tx_out });
                log::info!("client {} connected", id);
            }
        });

        Ok(NetServer {
            inbound: rx_inbound,
            new_clients: rx_new,
            clients: HashMap::new(),
        })
    }

    pub fn poll(&mut self) -> (Vec<PlayerId>, Vec<(PlayerId, Message)>) {
        let mut joined = Vec::new();
        while let Ok(nc) = self.new_clients.try_recv() {
            self.clients.insert(nc.id, nc.tx);
            joined.push(nc.id);
        }
        let mut msgs = Vec::new();
        while let Ok(m) = self.inbound.try_recv() {
            msgs.push(m);
        }
        (joined, msgs)
    }

    pub fn send_to(&self, id: PlayerId, msg: Message) {
        if let Some(tx) = self.clients.get(&id) {
            let _ = tx.send(msg);
        }
    }

    pub fn broadcast(&self, msg: &Message, except: Option<PlayerId>) {
        for (id, tx) in &self.clients {
            if Some(*id) == except {
                continue;
            }
            let _ = tx.send(msg.clone());
        }
    }

    /// Broadcast to clients matching the predicate. Used for distance-based
    /// interest management — voxel edits and position updates only fan out
    /// to players close enough to care.
    pub fn broadcast_filter<F: FnMut(PlayerId) -> bool>(&self, msg: &Message, mut keep: F) {
        for (id, tx) in &self.clients {
            if keep(*id) {
                let _ = tx.send(msg.clone());
            }
        }
    }

    pub fn drop_client(&mut self, id: PlayerId) {
        self.clients.remove(&id);
    }
}
